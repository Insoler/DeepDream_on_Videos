import argparse
import os




parser=argparse.ArgumentParser()
parser.add_argument('path',type=str)

args=parser.parse_args()
source_path=args.path 

print(source_path)


videos=os.listdir(source_path)

framepath= os.mkdir(source_path+'/frames')

for video in videos:
	os.system('ffmpeg -i {}/{} -r 1/{}  {}/frames/%03d.jpg'.format(source_path,video,fps,source_path))

