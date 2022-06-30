# torch.nn.CubicHermiteInterpolation
CubicHermite interpolators written in PyTorch. These funcitons inherit nn.Module, are autograd-compatible, and can run inside neural networks on GPUs.

The original dimensions must be specified for every axis when defining the interpolator module. The desired dimensions are required inputs to the chinterp.interp() function call. This allows the interpolator to resample, crop, and interpolate with both uniform and non-uniform sampling schemes. 

Multi-dimensional data is interpolated one dimension at a time starting with the last dimension and moving inward. The dimensionality of the interpolator refers to the dimensionality of the data. CubicHermiteSplines1d accepts n-dimensional input data but will only interpolate along dim=-1.

Currently, this family of interpolators is based on cubic Hermite interpolation and includes: splines, akima, and modified akima. There are implementations for 1D, 2D, 3D, and 4D data.

If you have ideas, questions, or contributions, please post in the issues section.

If this repository is useful for your work, please cite this repo to improve visibility.
