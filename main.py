import argparse
import json

from algo import *
from coworking import Coworking

def main():
    parser = argparse.ArgumentParser(description='Coworking Spaces Recommendation System')
    parser.add_argument('--data', type=str, default='data/coworkings.json', help='path to coworkings data')
    parser.add_argument('--algo', type=str, default='euclid', help='algorithm to use',
                        choices=['euclid-pro', 'euclid', 'hamming', 'cluster'])
    parser.add_argument('coworking', type=int, help='coworking id to give recommendations for')
    parser.add_argument('--n', type=int, default=4, help='number of recommendations to give')
    args = parser.parse_args()

    input_file = open(args.data)
    json_data = json.load(input_file)
    data = Coworking.schema().load(json_data, many=True)

    if args.algo == 'euclid-pro':
        recommendations, distances = euclidian_pro.recommend(data, args.coworking, args.n)
        print('Recommended Coworkings:')
        for coworking, distance in zip(recommendations, distances):
            print(f'- {coworking}, Distance: {distance}')
    else:
        if args.algo == 'euclid':
            algo = EuclideanRecommender(data)
        elif args.algo == 'hamming':
            algo = HammingRecommender(data)
        elif args.algo == 'cluster':
            algo = ClusteringRecommender(data)

        algo.fit()
        recommendations = algo.recommend(args.coworking, args.n)
        print(recommendations)

if __name__ == '__main__':
    main()