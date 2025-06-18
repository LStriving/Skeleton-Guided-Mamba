import os
import json
import argparse


actionDict={
  "OralDelivery":1,
  "SoftPalateLift":2,
  "HyoidExercise":3,
  "UESOpen":4,
  "ThroatSwallow":5,
  "ThroatTransport":6,
  "LaryngealVestibuleClosure":7
}
def main(args):
    with open(args.json, 'r')as f:
        data = json.load(f)
    
    data_name = os.path.basename(args.json).split('.json')[0]
    with open(args.txt, 'w')as f:
        for record in data:
            row = [data_name, record['segment'][0], record['segment'][1], actionDict[record['label']], record['score']]
            row = [str(i) for i in row]
            f.write(' '.join(row))
            f.write("\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--json",type=str)
    parser.add_argument("--txt",type=str)
    args = parser.parse_args()
    main(args)