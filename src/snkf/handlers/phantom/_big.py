"""Functions to generate phantom from bezier description."""

import json
import os
import numpy as np
import scipy as sp
from skimage.transform import resize
from pathlib import Path
import matplotlib.path as mpltPath
from importlib.resources import files

NUMBA_AVAILABLE = True
try:
    import numba
except ImportError:
    NUMBA_AVAILABLE = False


def _inside_poly(points: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    path = mpltPath.Path(vertices)
    mask = path.contains_points(points)
    return mask


def _is_in_triangle(
    pts: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> np.ndarray:
    """Check if pt is in the triangle defined by v1,v2, v3."""
    a = (pts[:, 0] - y[0]) * (x[1] - y[1]) - (x[0] - y[0]) * (pts[:, 1] - y[1]) < 0
    b = (pts[:, 0] - z[0]) * (y[1] - z[1]) - (y[0] - z[0]) * (pts[:, 1] - z[1]) < 0
    c = (pts[:, 0] - x[0]) * (z[1] - x[1]) - (z[0] - x[0]) * (pts[:, 1] - x[1]) < 0
    return (a == b) & (b == c)


def _is_in_triangle_mplt(
    pts: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> np.ndarray:
    """Check if points are in triangle using check for polygons."""
    return _inside_poly(pts, np.vstack([x, y, z]))


if NUMBA_AVAILABLE:
    is_in_triangle = numba.njit("f4[:,:],f4[:],f4[:],f4[:]")(_is_in_triangle)
else:
    is_in_triangle = _is_in_triangle_mplt


def inside_bezier_region(
    controls: np.ndarray, X: np.ndarray, Y: np.ndarray
) -> np.ndarray:
    """Return a map defining the inner of a Bézier region.

    Parameters
    ----------
    controls
        (N,2) array of point describing the controls points of a
        closed quadratic spline.
    X
        X grid coordinate
    Y
        Y grid coordinate

    Returns
    -------
    np.ndarray
        Boolean array masking the inner region.

    Notes
    -----
    This function was adapted from the code of Guerquin-Kern_2012.
    """
    # The quadratic bézier poly-line is defined by a list of controls points.
    # at the midway of two control point lies a node of the poly-line.
    controls = np.array(controls, dtype="float32")
    r = 0.5 * (np.roll(controls, [1, 1]) + np.roll(controls, [2, 2]))
    c = np.roll(controls, [1, 1])
    rp1 = np.roll(r, [-1, -1])
    # defines the curvature of each poly-line.
    beta = c - np.roll(c, [1, 1])
    gamma = rp1 + r - 2 * c

    a = beta[:, 0] * gamma[:, 1] - beta[:, 1] * gamma[:, 0]

    all_points = np.empty((X.size, 2), dtype=np.float32)
    all_points[:, 0] = X.flatten()
    all_points[:, 1] = Y.flatten()
    # A bezier curve is contains in the convex hull of its control points.
    # So only the point that lies in the hull will be tested further down.
    hull = sp.spatial.ConvexHull(controls)
    inside_hull = _inside_poly(all_points, controls[hull.vertices, :])
    points_in_hull = all_points[inside_hull]
    inside = _inside_poly(points_in_hull, r)

    # for each bézier curve of the polyline
    for i in range(len(a)):
        #: ind = find(inpoly(X,Y,[r(i,1) c(i,1) rp1(i,1)],[r(i,2) c(i,2) rp1(i,2)]));
        # consider the points inside the triangle node - control - node
        ind = np.argwhere(is_in_triangle(points_in_hull, r[i], c[i], rp1[i]))
        #: b = -(X(ind)-r(i,1))*gamma(i,2)+(Y(ind)-r(i,2))*gamma(i,1);
        #: d = -(X(ind)-r(i,1))*beta(i,2)+(Y(ind)-r(i,2))*beta(i,1);
        tmp = points_in_hull[ind, :] - r[i]
        b = -tmp[..., 0] * gamma[i, 1] + tmp[..., 1] * gamma[i, 0]
        d = -tmp[..., 0] * beta[i, 1] + tmp[..., 1] * beta[i, 0]
        #: map(ind(b.^2<a(i)*d)) = (a(i)>=0);
        cond = b**2 < (a[i] * d)
        inside[ind[cond]] = a[i] >= 0
        # a>=0 for outward-pointing triangles: add the interior points
        # a<0 for inside-pointing triangles: remove the exterior points
    inside_hull[np.argwhere(inside_hull)[~inside]] = 0
    return inside_hull.reshape(X.shape)


def _get_phantom_data(phantom_data_name: str | os.PathLike) -> list[dict]:
    roi_idx = None
    if phantom_data_name == "big":
        location = files("snkf.handlers.phantom") / "big_phantom_data.json"
    elif isinstance(phantom_data_name, os.PathLike):
        location = Path(phantom_data_name)
    elif "big_roi" in phantom_data_name:
        roi_idx = phantom_data_name.split("-")[-1]
        location = files("snkf.handlers.phantom") / "big_phantom_roi.json"

    with location.open() as f:
        phantom_data = json.load(f)
    if roi_idx and roi_idx.isnumeric():
        phantom_data[int(roi_idx)]

    return phantom_data


def raster_phantom(
    shape: int | tuple[int, ...],
    phantom_data: str | dict | list[dict] | os.PathLike = "big",
    weighting: str = "rel",
    medical_view: bool = True,
) -> np.ndarray:
    """Rasterize a 2D phantom using data specified in json file.

    Parameters
    ----------
    shape
        2D shape of the phantom
    phantom_data , optional
        List of dict describing a region, or path to json file.
        By default this build the phantom from the BIG group.
    weighting
        how should the contrast be computed.
        - "rel" : the contrast of the region is added relatively (+=)
        - "abs" : the constrast of the region is set absolutely (=)
        - "label": each region gets an integer labelling.
    medical_view
        If true the spatial axis are flipped to get the classical anatomical view.

    Returns
    -------
    np.ndarray
        2D phantom.
    """
    if isinstance(shape, int):
        shape = (shape,) * 2
    if isinstance(phantom_data, (str, Path)):
        phantom_data = _get_phantom_data(phantom_data)
    elif isinstance(phantom_data, dict):
        phantom_data = [phantom_data]
    elif not isinstance(phantom_data, list):
        raise ValueError("phantom data is not usable.")

    im = np.zeros(shape)
    X, Y = np.meshgrid(
        np.arange(-np.floor(shape[0] / 2), np.floor((shape[0]) / 2)),
        np.arange(-np.floor(shape[0] / 2), np.floor((shape[0]) / 2)),
    )
    X = np.float32(X / shape[0])
    Y = np.float32(Y / shape[1])
    label = 1
    for region in phantom_data:
        if region["type"] == "bezier":
            mask = inside_bezier_region(region["control"], X, Y)
        elif region["type"] == "ellipse":
            ct = np.cos(region["angle"])
            st = np.sin(region["angle"])
            x1 = X - region["center"][0]
            x2 = Y - region["center"][1]
            u1 = 2 / region["width"][0] * (ct * x1 + st * x2)
            u2 = 2 / region["width"][1] * (-st * x1 + ct * x2)
            mask = np.sqrt(u1**2 + u2**2) <= 1
        else:
            raise ValueError("Unsupported region type.")
        if weighting == "label":
            im[mask] = label
            label += 1
        elif weighting == "rel":
            im[mask] += region["weight"]
        elif weighting == "abs":
            im[mask] = region["weight"]
        else:
            raise ValueError("Unsupported weighted type")
    if medical_view is True:
        im = im.T
    return im


def generate_phantom(
    shape: int | tuple[int, ...],
    raster_osf: int = 4,
    phantom_data: str = "big",
    anti_aliasing: bool = True,
) -> np.ndarray:
    """
    Generate a phantom at the correct shape.

    For best results a high resolution phantom is first rasterize
    and downsampled at the correct shape.

    Parameters
    ----------
    shape
        target 2D-Shape
    raster_osf
        oversampling factor for the rasterization.
    phantom_data
        phantom definition, see raster_phantom for complete description.
    anti_aliasing
        If True, the high resolution phantom is smooth with a gaussian kernel
        prior to the downsizing.

    Returns
    -------
    np.ndarray

    Generated phantom

    See Also
    --------
    raster_phantom
        Method to raster at a defined resolution.
    sklearn.transform.resize
        Resize a 2D image
    """
    if isinstance(shape, int):
        shape = (shape,) * 2
    im_big = raster_phantom(
        tuple(np.array(shape) * raster_osf), phantom_data=phantom_data
    )
    if raster_osf == 1:
        return im_big
    im_final = resize(im_big, shape, anti_aliasing=anti_aliasing)
    return im_final
