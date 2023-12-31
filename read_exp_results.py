import os


class File(object):
    def __init__(self, file_path):
        self.file_path = file_path
        with open(file_path, 'w') as f:
            f.write('')
            f.close()

    def write(self, content):
        with open(self.file_path, 'a') as f:
            f.write(content)
            f.close()


def get_last_acc(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            lines = f.readlines()
            line = lines[-2]
            if "{'epoch': 50, 'eval_acc':" in line or "{'epoch': 120, 'eval_acc':" in line or "{'epoch': 200, 'eval_acc':" in line:
                # 提取字典部分
                start_index = line.find("{")  # 找到字典的起始位置
                end_index = line.find("}")  # 找到字典的结束位置
                dict_str = line[start_index:end_index + 1]  # 提取字典字符串

                # 将字典字符串转换为字典对象
                import ast
                data_dict = ast.literal_eval(dict_str)

                # 提取eval_acc的值
                eval_acc = data_dict['eval_acc']
                return eval_acc
            else:
                return None
    else:
        return None


def get_result(path):
    if os.path.exists(path):
        files = os.listdir(path)
        files.sort(reverse=True)
        for file in files:
            result = get_last_acc(os.path.join(path, file))
            if result is not None:
                return result
        return 'error'
    else:
        return None


def main(exp_root, result_file):
    file = File(result_file)
    file.write('dataset,method,sym,sym,sym,sym,sym,asym,asym,asym,asym\n')
    file.write(',,0.0,0.2,0.4,0.6,0.8,0.1,0.2,0.3,0.4\n')
    for dataset in ['mnist', 'cifar10', 'cifar100', 'animal10n', 'food101', 'clothing1m']:
        if dataset in ['animal10n', 'food101', 'clothing1m']:
            methods = os.listdir(os.path.join(exp_root, dataset))
        else:
            methods = os.listdir(os.path.join(exp_root, dataset, 'sym'))
        methods.sort()
        for method in methods:
            file.write('{},{}'.format(dataset, method))
            if dataset in ['animal10n', 'food101', 'clothing1m']:
                result = get_result(os.path.join(exp_root, dataset, method))
                print(os.path.join(exp_root, dataset, method), ":", result)
                file.write(',{},,,,,,,,'.format(result))
            else:
                for noise_type in ['sym', 'asym']:
                    if noise_type == 'sym':
                        noise_rates = [0.0, 0.2, 0.4, 0.6, 0.8]
                    else:
                        noise_rates = [0.1, 0.2, 0.3, 0.4]
                    for noise_rate in noise_rates:
                        result = get_result(os.path.join(exp_root, dataset, noise_type, method, 'n{}'.format(noise_rate)))
                        print(os.path.join(exp_root, dataset, noise_type, method, 'n{}'.format(noise_rate)), ":", result)
                        file.write(',{}'.format(result))
            file.write('\n')


if __name__ == '__main__':
    root = '/home/fangzh21/code/Active-Negative-Loss/experiment'
    file = 'result.csv'
    main(root, file)

