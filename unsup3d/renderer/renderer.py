# coding: UTF-8
import torch
from torch import nn
import math
# import neural_renderer as nr
from .utils import *
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras,
    OpenGLOrthographicCameras,
    PointLights,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor
)

"""
    @date:    2020.06.22
    @author:  daiheng.gao
    @target:  replace neural_renderer with pytorch3d soft rasterizer.
    
    
    @date:    2020.06.23
    @author:  daiheng.gao
    @target:  use point cloud renderer.
    @readme:  revise get_face_idx() & get_textures_from_im() to the resolution of image itself to compatible with pytorch3d.
"""

EPS = 1e-7


class Renderer(nn.Module):
    def __init__(self, cfgs):
        super(Renderer, self).__init__()

        self.device = cfgs.get('device', 'cuda:0')
        self.image_size = cfgs.get('image_size', 64)
        self.min_depth = cfgs.get('min_depth', 0.9)
        self.max_depth = cfgs.get('max_depth', 1.1)
        self.rot_center_depth = cfgs.get('rot_center_depth', (self.min_depth+self.max_depth)/2)
        # todo: FoV (Field of View) was set to be an fixed value of 10 degree (according to the paper).
        self.fov = cfgs.get('fov', 10)
        self.tex_cube_size = cfgs.get('tex_cube_size', 2)
        self.renderer_min_depth = cfgs.get('renderer_min_depth', 0.1)
        self.renderer_max_depth = cfgs.get('renderer_max_depth', 10.)

        #### camera intrinsics
        #             (u)   (x)
        #    d * K^-1 (v) = (y)
        #             (1)   (z)

        ## renderer for visualization
        R = [[[1.,0.,0.],
              [0.,1.,0.],
              [0.,0.,1.]]]
        R = torch.FloatTensor(R).to(self.device)
        t = torch.zeros(1,3, dtype=torch.float32).to(self.device)

        ## todo: K is the camera intrinsic matrix.
        fx = (self.image_size-1)/2/(math.tan(self.fov/2 *math.pi/180))
        fy = (self.image_size-1)/2/(math.tan(self.fov/2 *math.pi/180))
        cx = (self.image_size-1)/2
        cy = (self.image_size-1)/2
        K = [[fx, 0., cx],
             [0., fy, cy],
             [0., 0., 1.]]
        K = torch.FloatTensor(K).to(self.device)
        self.inv_K = torch.inverse(K).unsqueeze(0)
        self.K = K.unsqueeze(0)

        ## todo: define Renderer.
        ## use renderer from pytorch3d.
        # fixme: znear and zfar is equivalent to the neural renderer default settings.
        # cameras = OpenGLOrthographicCameras(device=self.device, R=R, T=t, znear=0.01, zfar=100)
        cameras = OpenGLPerspectiveCameras(device=self.device, R=R, T=t,
                                           znear=self.renderer_min_depth,
                                           zfar=self.renderer_max_depth,
                                           fov=self.fov)

        raster_settings = PointsRasterizationSettings(
            image_size=self.image_size,
            radius=0.003,
            points_per_pixel=10,
            bin_size=None,
            max_points_per_bin=None
        )

        self.renderer = PointsRenderer(
            rasterizer=PointsRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            compositor=AlphaCompositor(
                composite_params=None
            )
        )

    def set_transform_matrices(self, view):
        self.rot_mat, self.trans_xyz = get_transform_matrices(view)

    def rotate_pts(self, pts, rot_mat):
        centroid = torch.FloatTensor([0.,0.,self.rot_center_depth]).to(pts.device).view(1,1,3)
        pts = pts - centroid  # move to centroid
        pts = pts.matmul(rot_mat.transpose(2,1))  # rotate
        pts = pts + centroid  # move back
        return pts

    def translate_pts(self, pts, trans_xyz):
        return pts + trans_xyz

    def depth_to_3d_grid(self, depth):
        b, h, w = depth.shape
        grid_2d = get_grid(b, h, w, normalize=False).to(depth.device)  # N x H x W x 2
        depth = depth.unsqueeze(-1)
        grid_3d = torch.cat((grid_2d, torch.ones_like(depth)), dim=3)
        grid_3d = grid_3d.matmul(self.inv_K.to(depth.device).transpose(2,1)) * depth
        return grid_3d

    def grid_3d_to_2d(self, grid_3d):
        b, h, w, _ = grid_3d.shape
        grid_2d = grid_3d / grid_3d[...,2:]
        grid_2d = grid_2d.matmul(self.K.to(grid_3d.device).transpose(2,1))[:,:,:,:2]
        WH = torch.FloatTensor([w-1, h-1]).to(grid_3d.device).view(1,1,1,2)
        grid_2d = grid_2d / WH *2.-1.  # normalize to -1~1
        return grid_2d

    def get_warped_3d_grid(self, depth):
        """
            unsup3d's warping method is to wrap the object itself rather than change the perspective camera like in pytorch3d.
        :param depth:
        :return:
        """
        b, h, w = depth.shape
        grid_3d = self.depth_to_3d_grid(depth).reshape(b,-1,3)
        grid_3d = self.rotate_pts(grid_3d, self.rot_mat)
        grid_3d = self.translate_pts(grid_3d, self.trans_xyz)
        return grid_3d.reshape(b,h,w,3) # return 3d vertices

    def get_inv_warped_3d_grid(self, depth):
        b, h, w = depth.shape
        grid_3d = self.depth_to_3d_grid(depth).reshape(b,-1,3)
        grid_3d = self.translate_pts(grid_3d, -self.trans_xyz)
        grid_3d = self.rotate_pts(grid_3d, self.rot_mat.transpose(2,1))
        return grid_3d.reshape(b,h,w,3) # return 3d vertices

    def get_warped_2d_grid(self, depth):
        grid_3d = self.get_warped_3d_grid(depth)
        grid_2d = self.grid_3d_to_2d(grid_3d)
        return grid_2d

    def get_inv_warped_2d_grid(self, depth):
        grid_3d = self.get_inv_warped_3d_grid(depth)
        grid_2d = self.grid_3d_to_2d(grid_3d)
        return grid_2d

    def warp_canon_depth(self, canon_depth):
        b, h, w = canon_depth.shape
        # grid_3d supposed to be the 3D mesh.
        grid_3d = self.get_warped_3d_grid(canon_depth).reshape(b,-1,3)
        faces = get_face_idx(b, h, w).to(canon_depth.device)
        # fixme: https://github.com/facebookresearch/pytorch3d/issues/202
        grid_3d = Pointclouds(points=grid_3d, features=faces)
        warped_depth = self.renderer(grid_3d).squeeze()
        # allow some margin out of valid range
        margin = (self.max_depth - self.min_depth) / 2
        warped_depth = warped_depth.clamp(min=self.min_depth-margin, max=self.max_depth+margin)
        return warped_depth

    def get_normal_from_depth(self, depth):
        b, h, w = depth.shape
        grid_3d = self.depth_to_3d_grid(depth)

        tu = grid_3d[:,1:-1,2:] - grid_3d[:,1:-1,:-2]
        tv = grid_3d[:,2:,1:-1] - grid_3d[:,:-2,1:-1]
        normal = tu.cross(tv, dim=3)

        zero = torch.FloatTensor([0,0,1]).to(depth.device)
        normal = torch.cat([zero.repeat(b,h-2,1,1), normal, zero.repeat(b,h-2,1,1)], 2)
        normal = torch.cat([zero.repeat(b,1,w,1), normal, zero.repeat(b,1,w,1)], 1)
        normal = normal / (((normal**2).sum(3, keepdim=True))**0.5 + EPS)
        return normal

    def render_yaw(self, im, depth, v_before=None, v_after=None, rotations=None, maxr=90, nsample=9, crop_mesh=None):
        b, c, h, w = im.shape
        grid_3d = self.depth_to_3d_grid(depth)

        if crop_mesh is not None:
            top, bottom, left, right = crop_mesh  # pixels from border to be cropped
            if top > 0:
                grid_3d[:,:top,:,1] = grid_3d[:,top:top+1,:,1].repeat(1,top,1)
                grid_3d[:,:top,:,2] = grid_3d[:,top:top+1,:,2].repeat(1,top,1)
            if bottom > 0:
                grid_3d[:,-bottom:,:,1] = grid_3d[:,-bottom-1:-bottom,:,1].repeat(1,bottom,1)
                grid_3d[:,-bottom:,:,2] = grid_3d[:,-bottom-1:-bottom,:,2].repeat(1,bottom,1)
            if left > 0:
                grid_3d[:,:,:left,0] = grid_3d[:,:,left:left+1,0].repeat(1,1,left)
                grid_3d[:,:,:left,2] = grid_3d[:,:,left:left+1,2].repeat(1,1,left)
            if right > 0:
                grid_3d[:,:,-right:,0] = grid_3d[:,:,-right-1:-right,0].repeat(1,1,right)
                grid_3d[:,:,-right:,2] = grid_3d[:,:,-right-1:-right,2].repeat(1,1,right)

        grid_3d = grid_3d.reshape(b,-1,3)
        im_trans = []

        # inverse warp
        if v_before is not None:
            rot_mat, trans_xyz = get_transform_matrices(v_before)
            grid_3d = self.translate_pts(grid_3d, -trans_xyz)
            grid_3d = self.rotate_pts(grid_3d, rot_mat.transpose(2,1))

        if rotations is None:
            rotations = torch.linspace(-math.pi/180*maxr, math.pi/180*maxr, nsample)
        for i, ri in enumerate(rotations):
            ri = torch.FloatTensor([0, ri, 0]).to(im.device).view(1,3)
            rot_mat_i, _ = get_transform_matrices(ri)
            grid_3d_i = self.rotate_pts(grid_3d, rot_mat_i.repeat(b,1,1))

            if v_after is not None:
                if len(v_after.shape) == 3:
                    v_after_i = v_after[i]
                else:
                    v_after_i = v_after
                rot_mat, trans_xyz = get_transform_matrices(v_after_i)
                grid_3d_i = self.rotate_pts(grid_3d_i, rot_mat)
                grid_3d_i = self.translate_pts(grid_3d_i, trans_xyz)

            textures = get_textures_from_im(im, tx_size=self.tex_cube_size)
            grid_3d_i = Pointclouds(points=grid_3d_i, features=textures)
            warped_images = self.renderer(grid_3d_i).clamp(min=-1., max=1.)
            # print("扭过来的图像大小", warped_images.shape)  torch.Size([8, 64, 64, 3])
            im_trans += [warped_images]
        return torch.stack(im_trans, 1)  # b x t x c x h x w
