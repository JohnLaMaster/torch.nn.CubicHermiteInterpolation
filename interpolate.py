'''
https://stackoverflow.com/questions/61616810/how-to-do-cubic-spline-interpolation-and-integration-in-pytorch
By: chausies
Nov 13, 2020 - Converted to 3D tensors for PyTorch NN compatability by Ing. John T LaMaster
Mar 16, 2021 - replaced remaining NumPy with PyTorch implementation
Jan 21, 2022 - Added the Cubic Hermite Modified Akima Interpolation which better handles undulations and over-/undershoots
May 23, 2022 - Added the 2D implementations

Modified Akima is prefered for general MRS applications as it can reliably handle spectra at various stages of processing
'''
import torch
import torch.nn as nn


__all__ = ['CubicHermiteSplines1d', 'CubicHermiteAkima1d', 'CubicHermiteMAkima1d', 
           'CubicHermiteSplines2d', 'CubicHermiteAkima2d', 'CubicHermiteMAkima2d',
           'CubicHermiteSplines3d', 'CubicHermiteAkima3d', 'CubicHermiteMAkima3d',
           'CubicHermiteSplines4d', 'CubicHermiteAkima4d', 'CubicHermiteMAkima4d']


'''
The 1D implementations accept N-d inputs, but treat them as a compilation of 1D signals and interpolates 
along dim=-1. The 2D implementations work the same way and infact perform 2x 1D interpolations: first in 
the x-direction and then the y-direction.
'''


class CubicHermite1d(nn.Module):
    def __init__(self, 
                 xaxis: torch.Tensor, 
                 signal: torch.Tensor):
        super().__init__()
        self.device = signal.device
        for _ in range(signal.ndim - xaxis.ndim): xaxis = xaxis.unsqueeze(0)
        self.x = xaxis.expand_as(signal).to(self.device).contiguous()
        self.y = signal.contiguous()
        self.m = (signal[...,1:] - signal[...,:-1]) / (self.x[...,1:] - self.x[...,:-1])

    @staticmethod
    def h_poly_helper(tt):
        out = torch.empty_like(tt)
        A = torch.tensor([
                          [1, 0, -3,  2],
                          [0, 1, -2,  1],
                          [0, 0,  3, -2],
                          [0, 0, -1,  1]
                         ], dtype=tt[-1].dtype, device=tt.device)

        for r in range(4):
            out[...,r,:] = A[r,0] * tt[...,0,:] + A[r,1] * tt[...,1,:] + \
                           A[r,2] * tt[...,2,:] + A[r,3] * tt[...,3,:]
        return out

    def h_poly(self, t):
        tt = torch.empty_like(t).unsqueeze(-2).repeat_interleave(4, dim=-2).to(self.device)
        tt[...,0,:] = torch.ones_like(tt[...,0,:])
        for i in range(1, 4):
            tt[...,i,:] = tt[...,i-1,:].clone() * t
        return self.h_poly_helper(tt)

    def interp(self, 
               xs: torch.Tensor) -> torch.Tensor:
        size, size0 = list([i for i in self.x.shape]), list([1 for _ in range(self.x.ndim)])
        for _ in range(self.x.ndim - xs.ndim): xs = xs.unsqueeze(0)
        for i in range(2, xs.ndim-1): size0[i] = xs.shape[i]
        size[-1] = -1
           
        xs = xs.expand(size).contiguous()
        I  = torch.searchsorted(self.x[...,1:-1].contiguous(), xs.contiguous()).repeat(size0)

        x  = torch.gather(self.x, -1, I)
        dx = torch.gather(self.x, -1, I+1) - x
        
        hh = self.h_poly((xs - x)/dx)
            
        return hh[...,0,:]*torch.gather(self.y,-1,I)   + hh[...,1,:]*torch.gather(self.m,-1,I)*dx + \
               hh[...,2,:]*torch.gather(self.y,-1,I+1) + hh[...,3,:]*torch.gather(self.m,-1,I+1)*dx

    
    
class CubicHermiteSplines1d(CubicHermite1d):
    def __init__(self, 
                 xaxis: torch.Tensor, 
                 signal: torch.Tensor):
        super().__init__(xaxis, signal)
        self.m = torch.cat([self.m[...,0].unsqueeze(-1), (self.m[...,1:] + self.m[...,:-1]) / 2, self.m[...,-1].unsqueeze(-1)], dim=-1)
    
    
class CubicHermiteAkima1d(CubicHermite1d):
    '''
    Cubic Hermite Modified Akima Interpolation
    https://blogs.mathworks.com/cleve/2019/04/29/makima-piecewise-cubic-interpolation/
    
    Written by: Ing. John T LaMaster, 2022
    
    References:
    - https://en.wikipedia.org/wiki/Cubic_Hermite_spline
    - https://de-m-wikipedia-org.translate.goog/wiki/Akima-Interpolation?_x_tr_sl=de&_x_tr_tl=en&_x_tr_hl=en&_x_tr_pto=sc
    - https://en.wikipedia.org/wiki/Akima_spline ***
    '''
    def __init__(self, 
                 xaxis: torch.Tensor, 
                 signal: torch.Tensor):
        super().__init__(xaxis, signal)
        d0 = 2 * self.m[...,0] - self.m[...,1]
        m0 = 2 * d0 - self.m[...,0]
        dn = 2 * self.m[...,-1] - self.m[...,-2]
        mn = 2 * dn - self.m[...,-1]
        self.m = torch.cat([m0.unsqueeze(-1), d0.unsqueeze(-1), self.m, dn.unsqueeze(-1), mn.unsqueeze(-1)], dim=-1)
        
        weights = torch.abs(self.m[...,1:] - self.m[...,:-1])
        weights1 = weights[...,:-2].clone()
        weights2 = weights[...,2:].clone()
        delta1 = self.m[...,1:-2].clone()
        delta2 = self.m[...,2:-1].clone()
        weights12 = weights1 + weights2 + 1e-6
        
        self.m = (weights2 / weights12) * delta1 + (weights1 / weights12) * delta2
        self.m[(weights12==0)] = 0.
    
    
class CubicHermiteMAkima1d(CubicHermite1d):
    '''
    Cubic Hermite Modified Akima Interpolation
    https://blogs.mathworks.com/cleve/2019/04/29/makima-piecewise-cubic-interpolation/
    
    Written by: Ing. John T LaMaster, 2022
    
    References:
    - https://en.wikipedia.org/wiki/Cubic_Hermite_spline
    - https://de-m-wikipedia-org.translate.goog/wiki/Akima-Interpolation?_x_tr_sl=de&_x_tr_tl=en&_x_tr_hl=en&_x_tr_pto=sc
    - https://en.wikipedia.org/wiki/Akima_spline ***
    '''
    def __init__(self, xaxis: torch.Tensor, signal: torch.Tensor):
        super().__init__(xaxis, signal)
        d0 = 2 * self.m[...,0] - self.m[...,1]
        m0 = 2 * d0 - self.m[...,0]
        dn = 2 * self.m[...,-1] - self.m[...,-2]
        mn = 2 * dn - self.m[...,-1]
        self.m = torch.cat([m0.unsqueeze(-1), d0.unsqueeze(-1), self.m, dn.unsqueeze(-1), mn.unsqueeze(-1)], dim=-1)
        
        weights = torch.abs(self.m[...,:-1] - self.m[...,1:]) + torch.abs((self.m[...,:-1] + self.m[...,1:])/2)
        weights1 = weights[...,:-2].clone()
        weights2 = weights[...,2:].clone()
        delta1 = self.m[...,1:-2].clone()
        delta2 = self.m[...,2:-1].clone()
        weights12 = weights1 + weights2 + 1e-6
        
        self.m = (weights2 / weights12) * delta1 + (weights1 / weights12) * delta2
        self.m[(weights12==0)] = 0.



class CubicHermite2d(nn.Module):
    '''
    Based on code above.
    Written by: Ing. John T LaMaster, May 2022
    '''
    def __init__(self, 
                 xaxis: torch.Tensor, 
                 yaxis: torch.Tensor, 
                 signal: torch.Tensor):
        super().__init__()
        self.device = signal.device
        for _ in range(signal.ndim - yaxis.ndim): yaxis = yaxis.unsqueeze(-1)
        for _ in range(signal.ndim - xaxis.ndim): xaxis = xaxis.unsqueeze(0)
        self.x = xaxis.expand_as(signal).to(self.device).contiguous()
        self.y = yaxis.expand_as(signal).to(self.device).contiguous()
        self.data = signal.contiguous()
        self.m_x = (signal[...,1:,:] - signal[...,:-1,:]) / (self.x[...,1:,:] - self.x[...,:-1,:])
        self.m_y = (signal[...,1:] - signal[...,:-1]) / (self.y[...,1:] - self.y[...,:-1])

    @staticmethod
    def h_poly_helper(tt):
        out = torch.empty_like(tt)
        A = torch.tensor([
                          [1, 0, -3,  2],
                          [0, 1, -2,  1],
                          [0, 0,  3, -2],
                          [0, 0, -1,  1]
                         ], dtype=tt[-1].dtype, device=tt.device)

        for r in range(4):
            out[...,r,:] = A[r,0] * tt[...,0,:] + A[r,1] * tt[...,1,:] + \
                           A[r,2] * tt[...,2,:] + A[r,3] * tt[...,3,:]
        return out

    def h_poly(self, t):
        tt = torch.empty_like(t).unsqueeze(-2).repeat_interleave(4, dim=-2).to(self.device)
        tt[...,0,:] = torch.ones_like(tt[...,0,:])
        for i in range(1, 4):
            tt[...,i,:] = tt[...,i-1,:].clone() * t
        return self.h_poly_helper(tt)

    def interp(self, 
               xs: torch.Tensor,
               ys: torch.Tensor) -> torch.Tensor:
        size   = list([i for i in self.x.shape])
        size_x = list([1 for _ in range(self.x.ndim)])
        size_y = list([1 for _ in range(self.y.ndim)])
        for _ in range(self.x.ndim - xs.ndim): xs = xs.unsqueeze(0)
        for _ in range(self.y.ndim - ys.ndim): ys = ys.unsqueeze(-1)
        for i in range(2, xs.ndim): size_x[i] = xs.shape[i-1]
        for i in range(2, ys.ndim): size_y[i] = ys.shape[i-2]
        size[-1], size[-2] = -1, -1
           
        xs = xs.expand(size).contiguous()
        ys = ys.expand(size).contiguous()

        I_x = torch.searchsorted(self.x[...,1:-1,:].contiguous(), xs.contiguous()).repeat(size_x)
        I_y = torch.searchsorted(self.y[...,:,1:-1].contiguous(), ys.contiguous()).repeat(size_y)

        x  = torch.gather(self.x, -1, I_x)
        dx = torch.gather(self.x, -1, I_x+1) - x
        
        y  = torch.gather(self.y, -1, I_y)
        dy = torch.gather(self.y, -1, I_y+1) - y

        u = (xs - x) / dx
        v = (ys - y) / dy

        h = torch.stack([self.h_poly(1 + 2*u) * (1 - u)**2, 
                         self.h_poly(u) * dx * (1 - u), 
                         self.h_poly(1 + 2*u - 2*v) * (1 - u)**2, 
                         self.h_poly(u - v) * dx * (1 - u)], dim=-3)

        I_y = I_y.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.m_y.shape[-1])
        I_x = I_x.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.m_x.shape[-2], 1)
           
        m = torch.stack([torch.gather(self.m_y, -2, I_y), 
                         torch.gather(self.m_y, -2, I_y+1), 
                         torch.gather(self.m_x, -1, I_x), 
                         torch.gather(self.m_x, -1, I_x+1)], dim=-3)

        return torch.einsum('ijklmn,ijkln->ijkmn', h, m).squeeze(-3)
    
    
class CubicHermiteSplines2d(CubicHermite2d):
    def __init__(self, 
                 xaxis: torch.Tensor, 
                 yaxis: torch.Tensor, 
                 signal: torch.Tensor):
        super().__init__(xaxis, yaxis, signal)
        self.m1 = torch.cat([self.m1[...,0].unsqueeze(-1), (self.m1[...,1:] + self.m1[...,:-1]) / 2, self.m1[...,-1].unsqueeze(-1)], dim=-1)
        self.m2 = torch.cat([self.m2[...,0].unsqueeze(-1), (self.m2[...,1:] + self.m2[...,:-1]) / 2, self.m2[...,-1].unsqueeze(-1)], dim=-1)
    
    
class CubicHermiteAkima2d(CubicHermite2d):
    '''
    Cubic Hermite Modified Akima Interpolation
    https://blogs.mathworks.com/cleve/2019/04/29/makima-piecewise-cubic-interpolation/
    
    Written by: Ing. John T LaMaster, 2022
    
    References:
    - https://en.wikipedia.org/wiki/Cubic_Hermite_spline
    - https://de-m-wikipedia-org.translate.goog/wiki/Akima-Interpolation?_x_tr_sl=de&_x_tr_tl=en&_x_tr_hl=en&_x_tr_pto=sc
    - https://en.wikipedia.org/wiki/Akima_spline ***
    '''
    def __init__(self, 
                 xaxis: torch.Tensor, 
                 yaxis: torch.Tensor, 
                 signal: torch.Tensor):
        super().__init__(xaxis, yaxis, signal)
        self.m1 = self.calculate_m(self.m1)
        self.m2 = self.calculate_m(self.m2)
        
    def calculate_m(self, m):
        d0 = 2 * m[...,0] - m[...,1]
        m0 = 2 * d0 - m[...,0]
        dn = 2 * m[...,-1] - m[...,-2]
        mn = 2 * dn - m[...,-1]
        m = torch.cat([m0.unsqueeze(-1), d0.unsqueeze(-1), self.m, dn.unsqueeze(-1), mn.unsqueeze(-1)], dim=-1)
        
        weights = torch.abs(m[...,1:] - m[...,:-1])
        weights1 = weights[...,:-2].clone()
        weights2 = weights[...,2:].clone()
        delta1 = m[...,1:-2].clone()
        delta2 = m[...,2:-1].clone()
        weights12 = weights1 + weights2 + 1e-6
        
        m = (weights2 / weights12) * delta1 + (weights1 / weights12) * delta2
        m[(weights12==0)] = 0.
        return m
    
class CubicHermiteMAkima2d(CubicHermite2d):
    '''
    Cubic Hermite Modified Akima Interpolation
    https://blogs.mathworks.com/cleve/2019/04/29/makima-piecewise-cubic-interpolation/
    
    Written by: Ing. John T LaMaster, 2022
    
    References:
    - https://en.wikipedia.org/wiki/Cubic_Hermite_spline
    - https://de-m-wikipedia-org.translate.goog/wiki/Akima-Interpolation?_x_tr_sl=de&_x_tr_tl=en&_x_tr_hl=en&_x_tr_pto=sc
    - https://en.wikipedia.org/wiki/Akima_spline ***
    '''
    def __init__(self, xaxis: torch.Tensor, yaxis: torch.Tensor, signal: torch.Tensor):
        super().__init__(xaxis, yaxis, signal)
        self.m1 = self.calculate_m(self.m1)
        self.m2 = self.calculate_m(self.m2)

    def calculate_m(self, m):
        d0 = 2 * m[...,0] - m[...,1]
        m0 = 2 * d0 - m[...,0]
        dn = 2 * m[...,-1] - m[...,-2]
        mn = 2 * dn - m[...,-1]
        m = torch.cat([m0.unsqueeze(-1), d0.unsqueeze(-1), m, dn.unsqueeze(-1), mn.unsqueeze(-1)], dim=-1)
        
        weights = torch.abs(m[...,:-1] - m[...,1:]) + torch.abs((m[...,:-1] + m[...,1:])/2)
        weights1 = weights[...,:-2].clone()
        weights2 = weights[...,2:].clone()
        delta1 = m[...,1:-2].clone()
        delta2 = m[...,2:-1].clone()
        weights12 = weights1 + weights2 + 1e-6
        
        m = (weights2 / weights12) * delta1 + (weights1 / weights12) * delta2
        m[(weights12==0)] = 0.
        return m

    
    
# class CubicHermite3d(nn.Module):
#     '''
#     Based on code above.
#     Written by: Ing. John T LaMaster, June 2022
#     Updated: May 2023
#     '''
#     def __init__(self, 
#                  xaxis: torch.Tensor, 
#                  yaxis: torch.Tensor, 
#                  zaxis: torch.Tensor, 
#                  signal: torch.Tensor):
#         super().__init__()
#         self.device = signal.device
        
#         assert(xaxis.ndim == yaxis.ndim & yaxis.ndim == zaxis.ndim)
#         if xaxis.ndim==2:
#             xaxis = xaxis.unsqueeze(-1).unsqueeze(-1)
#             yaxis = yaxis.unsqueeze(0).unsqueeze(-1)
#             zaxis = zaxis.unsqueeze(0).unsqueeze(0)
        
#         for _ in range(signal.ndim - xaxis.ndim): 
#             xaxis = xaxis.unsqueeze(0)
#             yaxis = yaxis.unsqueeze(0)
#             zaxis = zaxis.unsqueeze(0)
            
#         self.x = xaxis.expand_as(signal).to(self.device).contiguous()
#         self.y = yaxis.expand_as(signal).to(self.device).contiguous()
#         self.z = zaxis.expand_as(signal).to(self.device).contiguous()
#         self.data = signal.contiguous()
        
#         self.m_x = (signal[...,1:,:,:] - signal[...,:-1,:,:]) / (self.x[...,1:,:,:] - self.x[...,:-1,:,:])
#         self.m_y = (signal[...,:,1:,:] - signal[...,:,:-1,:]) / (self.y[...,:,1:,:] - self.y[...,:,:-1,:])
#         self.m_z = (signal[...,1:,:,:] - signal[...,:,:,:-1]) / (self.z[...,:,:,1:] - self.z[...,:,:,:-1])

#     @staticmethod
#     def h_poly_helper(tt):
#         out = torch.empty_like(tt)
#         A = torch.tensor([
#                           [1, 0, -3,  2],
#                           [0, 1, -2,  1],
#                           [0, 0,  3, -2],
#                           [0, 0, -1,  1]
#                          ], dtype=tt[-1].dtype, device=tt.device)

#         for r in range(4):
#             out[...,r,:] = A[r,0] * tt[...,0,:] + A[r,1] * tt[...,1,:] + \
#                            A[r,2] * tt[...,2,:] + A[r,3] * tt[...,3,:]
#         return out

#     def h_poly(self, t):
#         tt = torch.empty_like(t).unsqueeze(-2).repeat_interleave(4, dim=-2).to(self.device)
#         tt[...,0,:] = torch.ones_like(tt[...,0,:])
#         for i in range(1, 4):
#             tt[...,i,:] = tt[...,i-1,:].clone() * t
#         return self.h_poly_helper(tt)

#     def interp(self, 
#                xs: torch.Tensor,
#                ys: torch.Tensor) -> torch.Tensor:
#         size   = list([i for i in self.x.shape])
#         size_x = list([1 for _ in range(self.x.ndim)])
#         size_y = list([1 for _ in range(self.y.ndim)])
#         size_z = list([1 for _ in range(self.z.ndim)])
#         for _ in range(self.x.ndim - xs.ndim): xs = xs.unsqueeze(0)
#         for _ in range(self.y.ndim - ys.ndim): ys = ys.unsqueeze(-1)
#         for _ in range(self.z.ndim - zs.ndim): zs = zs.unsqueeze(-1)
#         for i in range(2, xs.ndim): size_x[i] = xs.shape[i-1]
#         for i in range(2, ys.ndim): size_y[i] = ys.shape[i-2]
#         size[-1], size[-2] = -1, -1
           
#         xs = xs.expand(size).contiguous()
#         ys = ys.expand(size).contiguous()
#         zs = zs.expand(size).contiguous()

#         I_x = torch.searchsorted(self.x[...,1:-1,:,:].contiguous(), xs.contiguous()).repeat(size_x)
#         I_y = torch.searchsorted(self.y[...,:,1:-1,:].contiguous(), ys.contiguous()).repeat(size_y)
#         I_z = torch.searchsorted(self.z[...,:,:,1:-1].contiguous(), zs.contiguous()).repeat(size_z)

#         x  = torch.gather(self.x, -1, I_x)
#         dx = torch.gather(self.x, -1, I_x+1) - x
        
#         y  = torch.gather(self.y, -1, I_y)
#         dy = torch.gather(self.y, -1, I_y+1) - y
        
#         z  = torch.gather(self.z, -1, I_z)
#         dz = torch.gather(self.z, -1, I_z+1) - z

#         u = (xs - x) / dx
#         v = (ys - y) / dy
#         w = (zs - z) / dz

#         h = torch.stack([self.h_poly(1 + 2*u) * (1 - u)**2, 
#                          self.h_poly(u) * dx * (1 - u), 
#                          self.h_poly(1 + 2*u - 2*v) * (1 - u)**2, 
#                          self.h_poly(u - v) * dx * (1 - u)], dim=-3)

#         I_y = I_y.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.m_y.shape[-1])
#         I_x = I_x.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.m_x.shape[-2], 1)
           
#         m = torch.stack([torch.gather(self.m_y, -2, I_y), 
#                          torch.gather(self.m_y, -2, I_y+1), 
#                          torch.gather(self.m_x, -1, I_x), 
#                          torch.gather(self.m_x, -1, I_x+1)], dim=-3)

#         return torch.einsum('ijklmn,ijkln->ijkmn', h, m).squeeze(-3)
    
class CubicHermite3d(nn.Module):
    '''
    Based on code above.
    Written by: Ing. John T LaMaster, June 2022
    '''
    def __init__(self, 
                 xaxis: torch.Tensor, 
                 yaxis: torch.Tensor, 
                 zaxis: torch.Tensor, 
                 signal: torch.Tensor):
        super().__init__()
        self.device = signal.device
        for _ in range(signal.ndim - xaxis.ndim): xaxis = xaxis.unsqueeze(0)
        for _ in range(signal.ndim - yaxis.ndim): yaxis = yaxis.unsqueeze(0)
        for _ in range(signal.ndim - zaxis.ndim): zaxis = zaxis.unsqueeze(0)
        self.x1 = xaxis.expand_as(signal).to(self.device).contiguous()
        self.x2 = yaxis.expand_as(signal).to(self.device).contiguous()
        self.x3 = zaxis.expand_as(signal).to(self.device).contiguous()
        self.y = signal.contiguous()
        self.m1 = (signal[...,1:] - signal[...,:-1]) / (self.x1[...,1:] - self.x1[...,:-1])
        signal = signal.transpose(-2,-1)
        self.m2 = (signal[...,1:] - signal[...,:-1]) / (self.x2[...,1:] - self.x2[...,:-1])
        signal = signal.transpose(-2,-1).transpose(-3,-1)
        self.m3 = (signal[...,1:] - signal[...,:-1]) / (self.x3[...,1:] - self.x3[...,:-1])

    @staticmethod
    def h_poly_helper(tt):
        out = torch.empty_like(tt)
        A = torch.tensor([
                          [1, 0, -3,  2],
                          [0, 1, -2,  1],
                          [0, 0,  3, -2],
                          [0, 0, -1,  1]
                         ], dtype=tt[-1].dtype, device=tt.device)

        for r in range(4):
            out[...,r,:] = A[r,0] * tt[...,0,:] + A[r,1] * tt[...,1,:] + \
                           A[r,2] * tt[...,2,:] + A[r,3] * tt[...,3,:]
        return out

    def h_poly(self, t):
        tt = torch.empty_like(t).unsqueeze(-2).repeat_interleave(4, dim=-2).to(self.device)
        tt[...,0,:] = torch.ones_like(tt[...,0,:])
        for i in range(1, 4):
            tt[...,i,:] = tt[...,i-1,:].clone() * t
        return self.h_poly_helper(tt)

    def interp1d(self, input, x0, m, xs):
        size, size0 = list([i for i in x0.shape]), list([1 for _ in range(x0.ndim)])
        for _ in range(x0.ndim - xs.ndim): xs = xs.unsqueeze(0)
        for i in range(2, xs.ndim-1): size0[i] = xs.shape[i]
        size[-1] = -1
           
        xs = xs.expand(size).contiguous()
        I  = torch.searchsorted(x0[...,1:-1].contiguous(), xs.contiguous()).repeat(size0)

        x  = torch.gather(x0, -1, I)
        dx = torch.gather(x0, -1, I+1) - x
        
        hh = self.h_poly((xs - x)/dx)
            
        return hh[...,0,:]*torch.gather(input,-1,I)   + hh[...,1,:]*torch.gather(m,-1,I)*dx + \
               hh[...,2,:]*torch.gather(input,-1,I+1) + hh[...,3,:]*torch.gather(m,-1,I+1)*dx

    def interp(self, 
               xs: torch.Tensor, 
               ys: torch.Tensor, 
               zs: torch.Tensor) -> torch.Tensor:
        out = self.interp1d(self.y, self.x1, self.m1, xs).transpose(-2,-1)
        out = self.interp1d(out, self.x2, self.m2, ys).transpose(-2,-1).transpose(-3,-1)
        return self.interp1d(out, self.x3, self.m3, zs).transpose(-3,-1)

    
    
class CubicHermiteSplines3d(CubicHermite3d):
    def __init__(self, 
                 xaxis: torch.Tensor, 
                 yaxis: torch.Tensor, 
                 zaxis: torch.Tensor, 
                 signal: torch.Tensor):
        super().__init__(xaxis, yaxis, zaxis, signal)
        self.m1 = torch.cat([self.m1[...,0].unsqueeze(-1), (self.m1[...,1:] + self.m1[...,:-1]) / 2, self.m1[...,-1].unsqueeze(-1)], dim=-1)
        self.m2 = torch.cat([self.m2[...,0].unsqueeze(-1), (self.m2[...,1:] + self.m2[...,:-1]) / 2, self.m2[...,-1].unsqueeze(-1)], dim=-1)
        self.m3 = torch.cat([self.m3[...,0].unsqueeze(-1), (self.m3[...,1:] + self.m3[...,:-1]) / 2, self.m3[...,-1].unsqueeze(-1)], dim=-1)
    
    
class CubicHermiteAkima3d(CubicHermite3d):
    '''
    Cubic Hermite Modified Akima Interpolation
    https://blogs.mathworks.com/cleve/2019/04/29/makima-piecewise-cubic-interpolation/
    
    Written by: Ing. John T LaMaster, 2022
    
    References:
    - https://en.wikipedia.org/wiki/Cubic_Hermite_spline
    - https://de-m-wikipedia-org.translate.goog/wiki/Akima-Interpolation?_x_tr_sl=de&_x_tr_tl=en&_x_tr_hl=en&_x_tr_pto=sc
    - https://en.wikipedia.org/wiki/Akima_spline ***
    '''
    def __init__(self, 
                 xaxis: torch.Tensor, 
                 yaxis: torch.Tensor, 
                 zaxis: torch.Tensor, 
                 signal: torch.Tensor):
        super().__init__(xaxis, yaxis, zaxis, signal)
        self.m1 = self.calculate_m(self.m1)
        self.m2 = self.calculate_m(self.m2)
        self.m3 = self.calculate_m(self.m3)
        
    def calculate_m(self, m):
        d0 = 2 * m[...,0] - m[...,1]
        m0 = 2 * d0 - m[...,0]
        dn = 2 * m[...,-1] - m[...,-2]
        mn = 2 * dn - m[...,-1]
        m = torch.cat([m0.unsqueeze(-1), d0.unsqueeze(-1), self.m, dn.unsqueeze(-1), mn.unsqueeze(-1)], dim=-1)
        
        weights = torch.abs(m[...,1:] - m[...,:-1])
        weights1 = weights[...,:-2].clone()
        weights2 = weights[...,2:].clone()
        delta1 = m[...,1:-2].clone()
        delta2 = m[...,2:-1].clone()
        weights12 = weights1 + weights2 + 1e-6
        
        m = (weights2 / weights12) * delta1 + (weights1 / weights12) * delta2
        m[(weights12==0)] = 0.
        return m    
    
class CubicHermiteMAkima3d(CubicHermite3d):
    '''
    Cubic Hermite Modified Akima Interpolation
    https://blogs.mathworks.com/cleve/2019/04/29/makima-piecewise-cubic-interpolation/
    
    Written by: Ing. John T LaMaster, 2022
    
    References:
    - https://en.wikipedia.org/wiki/Cubic_Hermite_spline
    - https://de-m-wikipedia-org.translate.goog/wiki/Akima-Interpolation?_x_tr_sl=de&_x_tr_tl=en&_x_tr_hl=en&_x_tr_pto=sc
    - https://en.wikipedia.org/wiki/Akima_spline ***
    '''
    def __init__(self, 
                 xaxis: torch.Tensor, 
                 yaxis: torch.Tensor, 
                 zaxis: torch.Tensor, 
                 signal: torch.Tensor):
        super().__init__(xaxis, yaxis, zaxis, signal)
        self.m1 = self.calculate_m(self.m1)
        self.m2 = self.calculate_m(self.m2)
        self.m3 = self.calculate_m(self.m3)

    def calculate_m(self, m):
        d0 = 2 * m[...,0] - m[...,1]
        m0 = 2 * d0 - m[...,0]
        dn = 2 * m[...,-1] - m[...,-2]
        mn = 2 * dn - m[...,-1]
        m = torch.cat([m0.unsqueeze(-1), d0.unsqueeze(-1), m, dn.unsqueeze(-1), mn.unsqueeze(-1)], dim=-1)
        
        weights = torch.abs(m[...,:-1] - m[...,1:]) + torch.abs((m[...,:-1] + m[...,1:])/2)
        weights1 = weights[...,:-2].clone()
        weights2 = weights[...,2:].clone()
        delta1 = m[...,1:-2].clone()
        delta2 = m[...,2:-1].clone()
        weights12 = weights1 + weights2 + 1e-6
        
        m = (weights2 / weights12) * delta1 + (weights1 / weights12) * delta2
        m[(weights12==0)] = 0.
        return m
    
    
class CubicHermite4d(nn.Module):
    '''
    Based on code above.
    Written by: Ing. John T LaMaster, June 2022
    '''
    def __init__(self, 
                 xaxis: torch.Tensor, 
                 yaxis: torch.Tensor, 
                 zaxis: torch.Tensor, 
                 waxis: torch.Tensor,
                 signal: torch.Tensor):
        super().__init__()
        self.device = signal.device
        for _ in range(signal.ndim - xaxis.ndim): xaxis = xaxis.unsqueeze(0)
        for _ in range(signal.ndim - yaxis.ndim): yaxis = yaxis.unsqueeze(0)
        for _ in range(signal.ndim - zaxis.ndim): zaxis = zaxis.unsqueeze(0)
        for _ in range(signal.ndim - waxis.ndim): waxis = waxis.unsqueeze(0)
        self.x1 = xaxis.expand_as(signal).to(self.device).contiguous()
        self.x2 = yaxis.expand_as(signal).to(self.device).contiguous()
        self.x3 = zaxis.expand_as(signal).to(self.device).contiguous()
        self.x4 = waxis.expand_as(signal).to(self.device).contiguous()
        self.y = signal.contiguous()
        self.m1 = (signal[...,1:] - signal[...,:-1]) / (self.x1[...,1:] - self.x1[...,:-1])
        signal = signal.transpose(-2,-1)
        self.m2 = (signal[...,1:] - signal[...,:-1]) / (self.x2[...,1:] - self.x2[...,:-1])
        signal = signal.transpose(-2,-1).transpose(-3,-1)
        self.m3 = (signal[...,1:] - signal[...,:-1]) / (self.x3[...,1:] - self.x3[...,:-1])
        signal = signal.transpose(-3,-1).transpose(-4,-1)
        self.m4 = (signal[...,1:] - signal[...,:-1]) / (self.x4[...,1:] - self.x4[...,:-1])

    @staticmethod
    def h_poly_helper(tt):
        out = torch.empty_like(tt)
        A = torch.tensor([
                          [1, 0, -3,  2],
                          [0, 1, -2,  1],
                          [0, 0,  3, -2],
                          [0, 0, -1,  1]
                         ], dtype=tt[-1].dtype, device=tt.device)

        for r in range(4):
            out[...,r,:] = A[r,0] * tt[...,0,:] + A[r,1] * tt[...,1,:] + \
                           A[r,2] * tt[...,2,:] + A[r,3] * tt[...,3,:]
        return out

    def h_poly(self, t):
        tt = torch.empty_like(t).unsqueeze(-2).repeat_interleave(4, dim=-2).to(self.device)
        tt[...,0,:] = torch.ones_like(tt[...,0,:])
        for i in range(1, 4):
            tt[...,i,:] = tt[...,i-1,:].clone() * t
        return self.h_poly_helper(tt)

    def interp1d(self, input, x0, m, xs):
        size, size0 = list([i for i in x0.shape]), list([1 for _ in range(x0.ndim)])
        for _ in range(x0.ndim - xs.ndim): xs = xs.unsqueeze(0)
        for i in range(2, xs.ndim-1): size0[i] = xs.shape[i]
        size[-1] = -1
           
        xs = xs.expand(size).contiguous()
        I  = torch.searchsorted(x0[...,1:-1].contiguous(), xs.contiguous()).repeat(size0)

        x  = torch.gather(x0, -1, I)
        dx = torch.gather(x0, -1, I+1) - x
        
        hh = self.h_poly((xs - x)/dx)
            
        return hh[...,0,:]*torch.gather(input,-1,I)   + hh[...,1,:]*torch.gather(m,-1,I)*dx + \
               hh[...,2,:]*torch.gather(input,-1,I+1) + hh[...,3,:]*torch.gather(m,-1,I+1)*dx

    def interp(self, 
               xs: torch.Tensor, 
               ys: torch.Tensor, 
               zs: torch.Tensor,
               ws: torch.Tensor) -> torch.Tensor:
        out = self.interp1d(self.y, self.x1, self.m1, xs).transpose(-2,-1)
        out = self.interp1d(out, self.x2, self.m2, ys).transpose(-2,-1).transpose(-3,-1)
        out = self.interp1d(out, self.x3, self.m3, zs).transpose(-3,-1).transpose(-4,-1)
        return self.interp1d(out, self.x4, self.m4, ws).transpose(-4,-1)

    
    
class CubicHermiteSplines4d(CubicHermite4d):
    def __init__(self, 
                 xaxis: torch.Tensor, 
                 yaxis: torch.Tensor, 
                 zaxis: torch.Tensor,
                 waxis: torch.Tensor,
                 signal: torch.Tensor):
        super().__init__(xaxis, yaxis, zaxis, signal)
        self.m1 = torch.cat([self.m1[...,0].unsqueeze(-1), (self.m1[...,1:] + self.m1[...,:-1]) / 2, self.m1[...,-1].unsqueeze(-1)], dim=-1)
        self.m2 = torch.cat([self.m2[...,0].unsqueeze(-1), (self.m2[...,1:] + self.m2[...,:-1]) / 2, self.m2[...,-1].unsqueeze(-1)], dim=-1)
        self.m3 = torch.cat([self.m3[...,0].unsqueeze(-1), (self.m3[...,1:] + self.m3[...,:-1]) / 2, self.m3[...,-1].unsqueeze(-1)], dim=-1)
        self.m4 = torch.cat([self.m3[...,0].unsqueeze(-1), (self.m3[...,1:] + self.m3[...,:-1]) / 2, self.m3[...,-1].unsqueeze(-1)], dim=-1)
    
    
class CubicHermiteAkima4d(CubicHermite4d):
    '''
    Cubic Hermite Modified Akima Interpolation
    https://blogs.mathworks.com/cleve/2019/04/29/makima-piecewise-cubic-interpolation/
    
    Written by: Ing. John T LaMaster, 2022
    
    References:
    - https://en.wikipedia.org/wiki/Cubic_Hermite_spline
    - https://de-m-wikipedia-org.translate.goog/wiki/Akima-Interpolation?_x_tr_sl=de&_x_tr_tl=en&_x_tr_hl=en&_x_tr_pto=sc
    - https://en.wikipedia.org/wiki/Akima_spline ***
    '''
    def __init__(self, 
                 xaxis: torch.Tensor, 
                 yaxis: torch.Tensor, 
                 zaxis: torch.Tensor, 
                 waxis: torch.Tensor, 
                 signal: torch.Tensor):
        super().__init__(xaxis, yaxis, zaxis, waxis, signal)
        self.m1 = self.calculate_m(self.m1)
        self.m2 = self.calculate_m(self.m2)
        self.m3 = self.calculate_m(self.m3)
        self.m4 = self.calculate_m(self.m4)
        
    def calculate_m(self, m):
        d0 = 2 * m[...,0] - m[...,1]
        m0 = 2 * d0 - m[...,0]
        dn = 2 * m[...,-1] - m[...,-2]
        mn = 2 * dn - m[...,-1]
        m = torch.cat([m0.unsqueeze(-1), d0.unsqueeze(-1), self.m, dn.unsqueeze(-1), mn.unsqueeze(-1)], dim=-1)
        
        weights = torch.abs(m[...,1:] - m[...,:-1])
        weights1 = weights[...,:-2].clone()
        weights2 = weights[...,2:].clone()
        delta1 = m[...,1:-2].clone()
        delta2 = m[...,2:-1].clone()
        weights12 = weights1 + weights2 + 1e-6
        
        m = (weights2 / weights12) * delta1 + (weights1 / weights12) * delta2
        m[(weights12==0)] = 0.
        return m    
    
    
class CubicHermiteMAkima4d(CubicHermite4d):
    '''
    Cubic Hermite Modified Akima Interpolation
    https://blogs.mathworks.com/cleve/2019/04/29/makima-piecewise-cubic-interpolation/
    
    Written by: Ing. John T LaMaster, 2022
    
    References:
    - https://en.wikipedia.org/wiki/Cubic_Hermite_spline
    - https://de-m-wikipedia-org.translate.goog/wiki/Akima-Interpolation?_x_tr_sl=de&_x_tr_tl=en&_x_tr_hl=en&_x_tr_pto=sc
    - https://en.wikipedia.org/wiki/Akima_spline ***
    '''
    def __init__(self, 
                 xaxis: torch.Tensor, 
                 yaxis: torch.Tensor, 
                 zaxis: torch.Tensor, 
                 waxis: torch.Tensor,
                 signal: torch.Tensor):
        super().__init__(xaxis, yaxis, zaxis, waxis, signal)
        self.m1 = self.calculate_m(self.m1)
        self.m2 = self.calculate_m(self.m2)
        self.m3 = self.calculate_m(self.m3)
        self.m4 = self.calculate_m(self.m4)

    def calculate_m(self, m):
        d0 = 2 * m[...,0] - m[...,1]
        m0 = 2 * d0 - m[...,0]
        dn = 2 * m[...,-1] - m[...,-2]
        mn = 2 * dn - m[...,-1]
        m = torch.cat([m0.unsqueeze(-1), d0.unsqueeze(-1), m, dn.unsqueeze(-1), mn.unsqueeze(-1)], dim=-1)
        
        weights = torch.abs(m[...,:-1] - m[...,1:]) + torch.abs((m[...,:-1] + m[...,1:])/2)
        weights1 = weights[...,:-2].clone()
        weights2 = weights[...,2:].clone()
        delta1 = m[...,1:-2].clone()
        delta2 = m[...,2:-1].clone()
        weights12 = weights1 + weights2 + 1e-6
        
        m = (weights2 / weights12) * delta1 + (weights1 / weights12) * delta2
        m[(weights12==0)] = 0.
        return m
    
