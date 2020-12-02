import argparse
import pprint

parser = argparse.ArgumentParser(description="")
parser.add_argument('--n', type=int, default=1000, help="number of data points")
parser.add_argument('--d', type=int, default=100, help="feature dimension")


if __name__ == "__main__":
    args = parser.parse_args()
    
    print("Gengerating dataset with args:")
    pprint.pprint(args)

    

