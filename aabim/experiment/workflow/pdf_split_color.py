import statistics

import fitz  # PyMuPDF
from PIL import Image, ImageChops
import io
from tqdm import tqdm
import numpy as np

def is_color(image, threshold=10):
    """Determine if an image is color or grayscale."""
    img = Image.open(io.BytesIO(image))
    img = img.convert('RGB')
    pixels = img.getdata()
    if any(np.std(pixels, axis=1) > 20):  # Color image check
        return True
    else:
        return False
    # """Determine if an image is color or grayscale, with a tolerance for minor variations."""
    # img = Image.open(io.BytesIO(image))
    # img_rgb = img.convert('RGB')
    # img_gray = img.convert('L').convert('RGB')  # Convert to grayscale and back to RGB
    #
    # # Calculate the difference between the original and grayscale images
    # diff = ImageChops.difference(img_rgb, img_gray)
    # diff_data = diff.getdata()
    #
    # breakpoint()
    #
    # # Count non-zero differences (indicative of color)
    # diff_pixels = sum(1 for pixel in diff_data if sum(pixel) > threshold)

    # If significant differences are found, it's a color image
    return diff_pixels > 0

# def is_color(image, stddev_threshold=15):
#     """Determine if an image is color or grayscale based on standard deviation of pixel values."""
#     img = Image.open(io.BytesIO(image))
#     img_rgb = np.array(img.convert('RGB'))
#
#     # Calculate the standard deviation of the color intensities
#     stddev = np.std(img_rgb, axis=(0, 1))
#
#     breakpoint()
#
#     # If the standard deviation is below the threshold, consider it grayscale
#     return np.any(stddev > stddev_threshold)

# def is_color(image, histogram_threshold=0.01):
#     """Determine if an image is grayscale based on histogram similarity."""
#     img = Image.open(io.BytesIO(image))
#     img_rgb = img.convert('RGB')
#
#     # Get histograms for each channel
#     r_hist = np.histogram(img_rgb.getchannel('R'), bins=256, range=(0, 255))[0]
#     g_hist = np.histogram(img_rgb.getchannel('G'), bins=256, range=(0, 255))[0]
#     b_hist = np.histogram(img_rgb.getchannel('B'), bins=256, range=(0, 255))[0]
#
#     # Normalize histograms
#     r_hist = r_hist / np.sum(r_hist)
#     g_hist = g_hist / np.sum(g_hist)
#     b_hist = b_hist / np.sum(b_hist)
#
#     # Calculate histogram differences
#     rg_diff = np.linalg.norm(r_hist - g_hist)
#     gb_diff = np.linalg.norm(g_hist - b_hist)
#     rb_diff = np.linalg.norm(r_hist - b_hist)
#
#     # If all differences are below the threshold, consider it grayscale
#     return (rg_diff < histogram_threshold) and (gb_diff < histogram_threshold) and (rb_diff < histogram_threshold)

def split_pdf(input_pdf, output_bw_pdf, output_color_pdf):
    """Split the input PDF into black-and-white and color PDFs."""
    doc = fitz.open(input_pdf)

    bw_pages = []
    color_pages = []

    for i in tqdm(range(len(doc))):
        page = doc.load_page(i)
        pix = page.get_pixmap()
        image = pix.tobytes()

        if is_color(image):
            color_pages.append(page)
        else:
            bw_pages.append(page)

    # Save black-and-white pages
    if bw_pages:
        bw_doc = fitz.open()
        for page in bw_pages:
            bw_doc.insert_pdf(doc, from_page=page.number, to_page=page.number)
        bw_doc.save(output_bw_pdf)
        bw_doc.close()

    # Save color pages
    if color_pages:
        color_doc = fitz.open()
        for page in color_pages:
            color_doc.insert_pdf(doc, from_page=page.number, to_page=page.number)
        color_doc.save(output_color_pdf)
        color_doc.close()

    doc.close()

# Usage example:
input_pdf = '/D/Documents/sail/Bowditch_Vol_1.pdf'
output_bw_pdf = 'output_bw.pdf'
output_color_pdf = 'output_color.pdf'
split_pdf(input_pdf, output_bw_pdf, output_color_pdf)