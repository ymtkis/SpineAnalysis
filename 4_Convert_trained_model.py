import os
from deepd3.model.builder import DeepD3_Model
from config_utils import load_config, cfg_get


if __name__ == "__main__":
    cfg = load_config("config.yaml")

    docker_base_path = cfg_get(cfg, ["docker_base_path"], None)
    if not docker_base_path:
        raise KeyError("config.yaml must define: docker_base_path")

    repeats = int(cfg_get(cfg, ["repeats"], 3))
    in_pattern = cfg_get(cfg, ["deepd3_convert", "in_weights_pattern"],
                        "DeepD3_model_s{i}.h5")
    out_pattern = cfg_get(cfg, ["deepd3_convert", "out_model_pattern"],
                        "DeepD3_None_model_s{i}.h5")

    for i in range(repeats):
        dm_none = DeepD3_Model(input_shape=(None, None, 1))

        in_filename = in_pattern.format(i=i)
        out_filename = out_pattern.format(i=i)

        in_path = os.path.join(docker_base_path, in_filename)
        out_path = os.path.join(docker_base_path, out_filename)

        if not os.path.isfile(in_path):
            print(f"[SKIP] weights not found: {in_path}")
            continue

        dm_none.load_weights(in_path)
        dm_none.save(out_path)
        print(f"[OK] saved: {out_path}")
