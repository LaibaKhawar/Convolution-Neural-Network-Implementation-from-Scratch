# 2D Convolution from Scratch

A manual NumPy implementation of 2D convolution and cross-correlation on grayscale images, with stride and padding options, applied to edge detection, blurring and sharpening kernels and compared visually with SciPy.

**Interactive demo:** [Run it in your browser](https://www.laibakhawar.com/work/convolution#demo)

Despite the repository name, this project implements the convolution operation that CNN layers are built on. It does not build or train a convolutional neural network.

## How it works

All code is in `Convolution.ipynb`. The core function is:

```python
apply_convolution(image, kernel=None, kernel_size=3, stride=1, padding='valid', mode='convolution')
```

- `mode='convolution'` flips the kernel horizontally and vertically before sliding it; `mode='correlation'` applies it as given.
- `padding='same'` zero-pads by `(k - 1) // 2` on each side; `padding='valid'` uses no padding.
- Output size is `(padded_size - k) // stride + 1` in each dimension.
- Each output value is `np.sum(region * kernel)` over the window, computed with two nested Python loops.
- If no kernel is passed, a random kernel with values in [-0.5, 0.5) is generated.

The notebook then applies a series of 3x3 kernels with `padding='same'` and `stride=1`, writing each result to a JPEG:

| Experiment | Output file |
| --- | --- |
| Vertical edge example `[[1,0,-1],[1,0,-1],[1,0,-1]]` | `output.jpg` |
| "Sobel" | `EdgeDetection_SobelOperator_output.jpg` |
| "Prewitt" | `EdgeDetection_PrewittOperator_output.jpg` |
| "Laplacian" | `EdgeDetection_LaplacianOperator_output.jpg` |
| Box blur (all 1/9) | `BlurringImg_BoxBlur_output.jpg` |
| Gaussian blur (1/16, 2/16, 4/16 weights) | `BlurringImg_GaussianBlur_output.jpg` |
| Sharpen `[[0,-1,0],[-1,5,-1],[0,-1,0]]` | `SharpeningImg_SharpeningKernel_output.jpg` |
| Symmetric kernel, convolution vs correlation | `SymmetricConvolution_output.jpg`, `SymmetricCorrelation_output.jpg` |
| Asymmetric kernel `[[1,2,1],[0,0,0],[-1,-2,-1]]`, convolution vs correlation | `AsymmetricConvolution_output.jpg`, `AsymmetricCorrelation_output.jpg` |

The last cell runs the manual function and `scipy.signal.convolve2d(img, kernel, mode='same', boundary='fill', fillvalue=0)` on the same image and kernel and displays both. The comparison is visual; no numeric difference is computed.

## Results

There are no numeric results. The notebook's recorded outputs are image plots of the convolution/correlation comparisons and of the manual vs SciPy outputs.

## Repository contents

- `Convolution.ipynb`: implementation and experiments.
- `requirements.txt`: jupyter, matplotlib, numpy, opencv-python, scipy.

## Running it

The notebook was last run with Python 3.9.

```bash
git clone https://github.com/LaibaKhawar/Convolution-Neural-Network-Implementation-from-Scratch.git
cd Convolution-Neural-Network-Implementation-from-Scratch
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook Convolution.ipynb
```

Every cell loads `GreyscaleImg.jpeg` from the working directory. That image is not in the repository, so place any grayscale (or colour, it is read as grayscale) JPEG there under that name first. The first two cells run `pip install` for OpenCV and can be skipped if `requirements.txt` is installed.

## Known limitations

- **Input image not committed.** `GreyscaleImg.jpeg` is missing, so the notebook cannot run as-is.
- **The "Prewitt" kernel is a copy of the "Sobel" one.** Both cells use `[[1,0,-1],[-2,0,2],[-1,0,1]]`, so the two outputs are identical.
- **Non-standard kernels.** That shared "Sobel" kernel has inconsistent signs between rows (standard Sobel x is `[[1,0,-1],[2,0,-2],[1,0,-1]]`), and the "Laplacian" `[[0,1,0],[-1,4,-1],[0,-1,0]]` is not the standard Laplacian `[[0,1,0],[1,-4,1],[0,1,0]]` (or its negation).
- Results are written with `cv2.imwrite` without rescaling, so negative or above-255 values from edge kernels are clipped in the saved files.
- The plots in the asymmetric comparison cell are titled "Symmetric kernel" because the display code was copied.
- The nested Python loops are slow on large images.

## Author

[Laiba Khawar](https://github.com/LaibaKhawar) · [LinkedIn](https://www.linkedin.com/in/laibakhawar/)
