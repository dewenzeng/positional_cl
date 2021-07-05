import numpy as np
import matplotlib.pyplot as plt

def pad_if_too_small(image, new_shape, pad_value=None):
  shape = tuple(list(image.shape))
  new_shape = tuple(np.max(np.concatenate((shape, new_shape)).reshape((2, len(shape))), axis=0))
  if pad_value is None:
      if len(shape) == 2:
          pad_value = image[0, 0]
      elif len(shape) == 3:
          pad_value = image[0, 0, 0]
      else:
          raise ValueError("Image must be either 2 or 3 dimensional")
  res = np.ones(list(new_shape), dtype=image.dtype) * pad_value
  start = np.array(new_shape) / 2. - np.array(shape) / 2.
  if len(shape) == 2:
      res[int(start[0]):int(start[0]) + int(shape[0]), int(start[1]):int(start[1]) + int(shape[1])] = image
  elif len(shape) == 3:
      res[int(start[0]):int(start[0]) + int(shape[0]), int(start[1]):int(start[1]) + int(shape[1]),
      int(start[2]):int(start[2]) + int(shape[2])] = image
  return res

def pad_and_or_crop(orig_data, new_shape, mode=None, coords=None):

  data = pad_if_too_small(orig_data, new_shape, pad_value=0)

  h, w = data.shape
  if mode == "centre":
    h_c = int(h / 2.)
    w_c = int(w / 2.)
  elif mode == "fixed":
    assert (coords is not None)
    h_c, w_c = coords
  elif mode == "random":
    h_c_min = int(new_shape[0] / 2.)
    w_c_min = int(new_shape[1] / 2.)

    if new_shape[0] % 2 == 1:
      h_c_max = h - 1 - int(new_shape[0] / 2.)
      w_c_max = w - 1 - int(new_shape[1] / 2.)
    else:
      h_c_max = h - int(new_shape[0] / 2.)
      w_c_max = w - int(new_shape[1] / 2.)

    h_c = np.random.randint(low=h_c_min, high=(h_c_max + 1))
    w_c = np.random.randint(low=w_c_min, high=(w_c_max + 1))

  h_start = h_c - int(new_shape[0] / 2.)
  w_start = w_c - int(new_shape[1] / 2.)
  data = data[h_start:(h_start + new_shape[0]), w_start:(w_start + new_shape[1])]

  return data, (h_c, w_c)

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    # img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg)
    else:
        # use this function if image is grayscale
        plt.imshow(npimg[0,:,:],'gray')
        # use this function if image is RGB
        # plt.imshow(np.transpose(npimg, (1, 2, 0)))
