import argparse
from glob import glob
import os
import torch
import logging
logger = logging.getLogger(__name__)


def main():
    args = arguments()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    checker = T3Checker(args.input, args.num_events)
    checker.load()
    checker.check()


def arguments():
    default_input = "/ceph/users/atuna/work/cms_tracking_ml/transformer-oc/data/pu200_t3/train/*.pt"
    parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=str, default=default_input, help="Input globbable file path")
    parser.add_argument("--num-events", type=int, default=10, help="Maximum number of events to plot")
    return parser.parse_args()


class T3Checker:

    def __init__(self, input_glob, num_events) -> None:
        self.input_glob = input_glob
        self.num_events = num_events


    def load(self) -> None:
        logger.info(f"Globbing files from {self.input_glob} ...")
        self.files = sorted(glob(self.input_glob))

        if len(self.files) == 0:
            raise FileNotFoundError(f"No files found for glob pattern: {self.input_glob}")
        elif len(self.files) > self.num_events:
            logger.info(f"Found {len(self.files)} files, requested {self.num_events}. Skipping the extra")
            self.files = self.files[:self.num_events]


    def check(self) -> None:
        logger.info(f"Checking {len(self.files)} files for redundancy ...")
        n_ok = 0
        for file in self.files:
            ok = True
            data = torch.load(file, weights_only=False)
            for feature in t3.MD_VARS:
                # MD feature_1 should be:
                #  - different from MD feature_0, and
                #  - identical to feature_2
                if not feature.endswith("_0"):
                    continue
                feature_base = feature.replace("_0", "")
                feature_0 = data.x[:, getattr(t3, feature_base + "_0")]
                feature_1 = data.x[:, getattr(t3, feature_base + "_1")]
                feature_2 = data.x[:, getattr(t3, feature_base + "_2")]
                check_01 = torch.allclose(feature_0, feature_1)
                check_12 = torch.allclose(feature_1, feature_2)
                ok = ok and (not check_01) and check_12

            if ok:
                logger.info(f"Redundancy check passed for {os.path.basename(file)}")
                n_ok += 1
            else:
                logger.warning(f"Redundancy check failed for {os.path.basename(file)}")
                break

        logger.info(f"{n_ok}/{len(self.files)} files have one completely redundant MD")

#
# Keeping track of which columns correspond to which t3 variables
#
class t3:
    T3_VARS = ["t3_pt", "t3_eta", "t3_phi"]
    LS_VARS = [
        "ls_dPhis_0", "ls_dPhiChanges_0", "ls_dAlphaInners_0", "ls_dAlphaOuters_0", "ls_dAlphaInnerOuters_0",
        "ls_dPhis_1", "ls_dPhiChanges_1", "ls_dAlphaInners_1", "ls_dAlphaOuters_1", "ls_dAlphaInnerOuters_1",
    ]
    MD_VARS = [
        "md_anchor_x_0", "md_anchor_y_0", "md_anchor_z_0", "md_other_x_0", "md_other_y_0", "md_other_z_0", "md_dphi_0", "md_dphichange_0", "md_dz_0",
        "md_anchor_x_1", "md_anchor_y_1", "md_anchor_z_1", "md_other_x_1", "md_other_y_1", "md_other_z_1", "md_dphi_1", "md_dphichange_1", "md_dz_1",
        "md_anchor_x_2", "md_anchor_y_2", "md_anchor_z_2", "md_other_x_2", "md_other_y_2", "md_other_z_2", "md_dphi_2", "md_dphichange_2", "md_dz_2",
        "md_anchor_x_3", "md_anchor_y_3", "md_anchor_z_3", "md_other_x_3", "md_other_y_3", "md_other_z_3", "md_dphi_3", "md_dphichange_3", "md_dz_3",
    ]
    ALL_VARS = T3_VARS + LS_VARS + MD_VARS
    pt = ALL_VARS.index("t3_pt")
    eta = ALL_VARS.index("t3_eta")
    phi = ALL_VARS.index("t3_phi")
    ls_dPhis_0 = ALL_VARS.index("ls_dPhis_0")
    ls_dPhiChanges_0 = ALL_VARS.index("ls_dPhiChanges_0")
    ls_dAlphaInners_0 = ALL_VARS.index("ls_dAlphaInners_0")
    ls_dAlphaOuters_0 = ALL_VARS.index("ls_dAlphaOuters_0")
    ls_dAlphaInnerOuters_0 = ALL_VARS.index("ls_dAlphaInnerOuters_0")
    ls_dPhis_1 = ALL_VARS.index("ls_dPhis_1")
    ls_dPhiChanges_1 = ALL_VARS.index("ls_dPhiChanges_1")
    ls_dAlphaInners_1 = ALL_VARS.index("ls_dAlphaInners_1")
    ls_dAlphaOuters_1 = ALL_VARS.index("ls_dAlphaOuters_1")
    ls_dAlphaInnerOuters_1 = ALL_VARS.index("ls_dAlphaInnerOuters_1")
    md_anchor_x_0 = ALL_VARS.index("md_anchor_x_0")
    md_anchor_y_0 = ALL_VARS.index("md_anchor_y_0")
    md_anchor_z_0 = ALL_VARS.index("md_anchor_z_0")
    md_other_x_0 = ALL_VARS.index("md_other_x_0")
    md_other_y_0 = ALL_VARS.index("md_other_y_0")
    md_other_z_0 = ALL_VARS.index("md_other_z_0")
    md_dphi_0 = ALL_VARS.index("md_dphi_0")
    md_dphichange_0 = ALL_VARS.index("md_dphichange_0")
    md_dz_0 = ALL_VARS.index("md_dz_0")
    md_anchor_x_1 = ALL_VARS.index("md_anchor_x_1")
    md_anchor_y_1 = ALL_VARS.index("md_anchor_y_1")
    md_anchor_z_1 = ALL_VARS.index("md_anchor_z_1")
    md_other_x_1 = ALL_VARS.index("md_other_x_1")
    md_other_y_1 = ALL_VARS.index("md_other_y_1")
    md_other_z_1 = ALL_VARS.index("md_other_z_1")
    md_dphi_1 = ALL_VARS.index("md_dphi_1")
    md_dphichange_1 = ALL_VARS.index("md_dphichange_1")
    md_dz_1 = ALL_VARS.index("md_dz_1")
    md_anchor_x_2 = ALL_VARS.index("md_anchor_x_2")
    md_anchor_y_2 = ALL_VARS.index("md_anchor_y_2")
    md_anchor_z_2 = ALL_VARS.index("md_anchor_z_2")
    md_other_x_2 = ALL_VARS.index("md_other_x_2")
    md_other_y_2 = ALL_VARS.index("md_other_y_2")
    md_other_z_2 = ALL_VARS.index("md_other_z_2")
    md_dphi_2 = ALL_VARS.index("md_dphi_2")
    md_dphichange_2 = ALL_VARS.index("md_dphichange_2")
    md_dz_2 = ALL_VARS.index("md_dz_2")
    md_anchor_x_3 = ALL_VARS.index("md_anchor_x_3")
    md_anchor_y_3 = ALL_VARS.index("md_anchor_y_3")
    md_anchor_z_3 = ALL_VARS.index("md_anchor_z_3")
    md_other_x_3 = ALL_VARS.index("md_other_x_3")
    md_other_y_3 = ALL_VARS.index("md_other_y_3")
    md_other_z_3 = ALL_VARS.index("md_other_z_3")
    md_dphi_3 = ALL_VARS.index("md_dphi_3")
    md_dphichange_3 = ALL_VARS.index("md_dphichange_3")
    md_dz_3 = ALL_VARS.index("md_dz_3")


if __name__ == "__main__":
    main()
