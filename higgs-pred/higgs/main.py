import hydra
import higgs.cmds


@hydra.main(config_path="..", config_name="higgs")
def main(cfg):
    arg = cfg[cfg.cmd] if cfg.cmd in cfg else None
    higgs.cmds.__getattribute__(cfg.cmd)(arg)


if __name__ == "__main__":
    main()
