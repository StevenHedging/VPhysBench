from physbench.baseline_api.endpoint import main
from physbench.baseline_plugins.wan22 import Wan22BaselinePlugin


if __name__ == "__main__":
    raise SystemExit(main(Wan22BaselinePlugin))
