import os

from yacs.config import CfgNode

from core.datasets.moving_mnist import make_db
from core.models.general_model import make_general_model
from core.tools.tester import Tester
from core.tools.trainer import Trainer
from core.tools.utils import (
    check_path,
    create_cfg,
    find_latest_experiment,
    set_seed,
)


def set_experiment_num(cfg: CfgNode) -> int:
    """
    Determine correct experiment number for logging.

    Config
    ------
    experiment_dir
    model.name
    model.resume_experiment_num
    model.resume
    """
    experiment_root = os.path.join(cfg.experiment_dir, cfg.model.name)
    check_path(experiment_root)

    save_experiment_path = os.path.join(experiment_root, "save")
    check_path(save_experiment_path)

    if cfg.model.resume:
        if cfg.model.resume_experiment_num == -1:
            experiment_num = int(sorted(os.listdir(save_experiment_path))[-1])
        else:
            experiment_num = cfg.model.resume_experiment_num
    else:
        experiment_num = find_latest_experiment(save_experiment_path) + 1

    return experiment_num


def check_directory():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root_dir)



def main():
    """
    Main entry point for loading config, datasets and then either starting training or 
    testing.

    Config
    ------
    system.seed
    model.resume
    model.resume_experiment_num

    Submodules
    ----------
    set_experiment_num
    make_db
    make_general_model
    Trainer
    run_study
    """
    check_directory()
    cfg = create_cfg()
    set_seed(cfg.system.seed)
    experiment_num = set_experiment_num(cfg)

    train_db = make_db(cfg, train=True)
    val_db = make_db(cfg, train=False)

    model = make_general_model(cfg)
    if cfg.model.resume and cfg.model.resume_experiment_num != -1:
        experiment_num = cfg.model.resume_experiment_num

    if not cfg.eval_only:

        trainer = Trainer(cfg, model, train_db, val_db, experiment_num)
        trainer.train_model()

    tester = Tester(cfg, model, val_db)
    tester.evaluate()


if __name__ == "__main__":
    main()
