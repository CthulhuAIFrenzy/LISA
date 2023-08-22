import deepspeed
from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", type=str, help="path to the desired checkpoint folder, e.g., path/checkpoint-12")
    parser.add_argument("output_file", type=str, help="path to the pytorch fp32 state_dict output file (e.g. path/checkpoint-12/pytorch_model.bin)")
    parser.add_argument("-t", "--tag", type=str, default=None, help="checkpoint tag used as a unique identifier for checkpoint. e.g., global_step1")
    parser.add_argument("-d", "--debug", action='store_true', help="enable debug")
    args = parser.parse_args()

    debug = args.debug

    convert_zero_checkpoint_to_fp32_state_dict(args.checkpoint_dir, args.output_file, tag=args.tag)