import os, sys

def trim_file(path, output_path, num_data):
  f = open(path, 'r')
  lines = f.readlines()
  f.close()
  lines = lines[:num_data]
  f = open(output_path, 'w')
  for line in lines:
    f.write(line)
  f.close()

if __name__ == '__main__':
  dir = sys.argv[1]
  workload = sys.argv[2]
  num_data = int(sys.argv[3])
  trim_file(os.path.join(dir, workload) + '-training.txt', os.path.join(dir, workload + '-small') + '-training.txt', num_data)
  trim_file(os.path.join(dir, workload) + '-testing.txt', os.path.join(dir, workload + '-small') + '-testing.txt', num_data)

