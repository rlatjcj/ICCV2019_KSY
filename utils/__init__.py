from .utils import *

def cnt_time(args, trainset, duration, batch_idx):
    residual = int(duration * args.steps - duration * (batch_idx + 1)) if args.steps else int(duration * (len(trainset)//args.batch_size) - duration * (batch_idx + 1))
    if residual <= 60:
        # only second
        return '{}s'.format(residual)
    elif 60 < residual <= 3600:
        # minute and second
        minute = int(residual // 60)
        second = residual % 60
        return '{}:{}'.format(minute, second)
    elif 3600 < residual <= 86400:
        # hour, minute, and second
        hour = int((residual / 60) // 60)
        minute = int((residual - hour * 3600) // 60)
        second = residual % 60
        return '{}:{}:{}'.format(hour, minute, second)
    elif residual > 86400:
        # day, hour, minute, and second
        day = int(((residual / 60) / 60) // 24)
        hour = int(((residual - day * 86400) / 60) // 60)
        minute = int((residual - day * 86400 - hour * 3600) // 60)
        second = residual % 60
        return '{}:{}:{}:{}'.format(day, hour, minute, second)
    else:
        raise ValueError()

def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=40): 
    formatStr = "{0:." + str(decimals) + "f}" 
    percent = formatStr.format(100 * (iteration / float(total))) 
    filledLength = int(round(barLength * iteration / float(total))) 
    bar = '#' * filledLength + '-' * (barLength - filledLength) 
    sys.stdout.write('\r%s/%s |%s| %s %s%s %s' % (iteration, total, bar, prefix, percent, '%', suffix)), 
    if iteration == total: 
        sys.stdout.write('\n') 
    sys.stdout.flush()