# Create Foveated Blur
Blur every pixel with a different kernel size based on the distance from the center of the image. The further away from the center, the larger the kernel size. This creates a foveated blur effect, where the center of the image is sharp and the edges are blurred.
Given a map of sigma values for each pixel, this code automatically creates a foveated blur effect on a series of images. Store the sigmas in a png and normalize it by dividing by the maximum sigma value and save the file as "SigmaPX-MAXVALUE.png". This transfomation will then be undone when applying the kernels.

## Installation
```shell
pip install -r requirements.txt
```

## Usage
```shell
python main.py --source <source_folder> --sigma_map <sigma_map-MAXVALUE.png>
```
Additional Parameters:
- `--suffix <suffix>`: Suffix to add to the output images. Default: "-s20"
- `--correct_source`: Correct the dynamic range in the source images the same way as done for the foveated images. The corrected image is saved with the suffix "-s0". Default: False
- `--num_processes <num_processes>`: Number of processes to use for parallel processing. Default: 9