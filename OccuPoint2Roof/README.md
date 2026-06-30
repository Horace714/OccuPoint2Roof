# OccuPoint2Roof

PyTorch implementation of end-to-end 3D building roof reconstruction from airborne LiDAR point clouds, with a focus on handling occluded regions in roof structures.

The pipeline consists of four stages: keypoint detection, cluster refinement, occluded completion, and edge attention. 
## Installation

**Requirements:** Python 3.8+, CUDA 11.x/12.x, PyTorch 2.x

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Install PyTorch with CUDA support (adjust cu118/cu124 to match your CUDA version)
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124

# 3. Build the C++/CUDA point cloud utilities
cd pc_util && python setup.py install && cd ..
```

## Training

```bash
python train.py \
    --data_path /path/to/dataset \
    --cfg_file ./model_cfg.yaml \
    --batch_size 128 \
    --epochs 200 \
    --lr 1e-3 \
    --gpu 0
```

Checkpoints and logs are saved to `output/<extra_tag>/ckpt/`.

## Testing

```bash
python test.py \
    --data_path /path/to/dataset \
    --cfg_file ./model_cfg.yaml \
    --batch_size 1 \
    --gpu 0 \
    --test_tag pts6
```

Results are written to `output/<test_tag>/test/`.


## Citation

If you use this code in your research, please cite the corresponding paper (to appear):

```bibtex
@article{occupoint2roof,
  title={OccuPoint2Roof: A Progressive Framework for Joint Completion and Reconstruction of 3D Building Roofs from Occluded LiDAR Point Clouds},
  author={Anonymous},
  year={2026}
}
```