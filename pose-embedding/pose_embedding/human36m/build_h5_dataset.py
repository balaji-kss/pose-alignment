from pose_embedding.human36m.Human36M import Human36M
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Pose embedding model training.')
    parser.add_argument(
        '--chunk-id', type=int, default=4, help='chunk id')     
    parser.add_argument(
        '--chunks', type=int, default=10, help='number of h5 files')     
    parser.add_argument(
        '--data-split', default="train", help='data split [train, test]')     
    parser.add_argument(
        '--dataset-path', default="/data/junhe/dataset/h36m", help='data set path')      
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    h36m = Human36M(data_split=args.data_split, dataset_path=args.dataset_path)
    h36m.load_keypoint3d()
    h36m.export_pairs_hdf5(hdf5_file=f'h36m_pair_{args.data_split}', 
                           chunk_id=args.chunk_id, chunks=args.chunks)