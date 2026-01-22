export DISPLAY=:2
export VGL_DISPLAY=:1
conda activate rlbench

vglrun -d :1 python rlbench/dataset_generator_single.py \
      --tasks put_groceries_in_cupboard \
      --save_path ./physics_data \
      --episodes_per_task 1 \
      --variations 1 \
      --image_size 128 128 \
      --renderer opengl3 \
      --max_attempts 1