import logging
import numpy as np
import torch
import datetime
from sklearn import metrics
import argparse
import os
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from scipy.io import loadmat
from tqdm import tqdm  # 添加tqdm导入A

from model_MIND import Mind
from dataloader import load_data_de, load_data_de1, load_data_vmd, \
    load_data_vmd_leave_one_subject_out, load_data_vmd1,load_data_denpendent,load_data_inde_yuan,load_data_inde1,load_data_inde2,load_data_denpendent1,load_data_denpendent2,load_data_denpendent3,load_data_inde3
from utils import CE_Label_Smooth_Loss, set_logging_config
from sklearn.cluster import KMeans
from utils import CenterLoss
np.set_printoptions(threshold=np.inf)



def auto_partition_channels(eeg_data, n_clusters=7, method='corr'):
    """
    根据EEG数据自动聚类通道，生成子图划分结果（每个通道属于哪个subgraph）。

    参数:
    ----
    eeg_data : ndarray, shape (channels, samples)y``y
        EEG数据，通道数 x 采样点 或通道数 x feature维度

    n_clusters : int
        要划分的子图数

    method : str, 'corr' 或 'euclidean'
        使用相关系数还是欧氏距离度量通道相似度

    返回:
    ----
    cluster_labels : ndarray, shape (channels,)
        每个通道对应的聚类标签(0 ~ n_clusters-1)

    subgraph_sizes : list
        每个子图的通道数
    """
    channels, samples = eeg_data.shape

    # 计算通道间距离/相似度
    if method == 'corr':
        # 相关系数矩阵 (channels x channels)
        corr_matrix = np.corrcoef(eeg_data)
        dist_matrix = 1 - corr_matrix  # 越小相似度越高
    else:
        # euclidean
        dist_matrix = np.zeros((channels, channels))
        for i in range(channels):
            for j in range(i + 1, channels):
                dist_val = np.linalg.norm(eeg_data[i] - eeg_data[j])
                dist_matrix[i, j] = dist_val
                dist_matrix[j, i] = dist_val

    # 距离 => 转化为相似度
    sim_features = 1 / (1 + dist_matrix)

    # 对通道做聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(sim_features)

    # 统计每个子图通道数
    subgraph_sizes = []
    for c in range(n_clusters):
        c_size = np.sum(cluster_labels == c)
        subgraph_sizes.append(int(c_size))

    return cluster_labels, subgraph_sizes




class Trainer(object):
    def __init__(self, args, subject_name):
        self.args = args
        self.subject_name = subject_name

    def train(self, data_and_label):
        logger = logging.getLogger("train")
        logger.propagate = False  # 防止日志输出干扰进度条

        # 添加数据集大小检查
        print(f"训练集形状: x_tr={data_and_label['x_tr'].shape}, y_tr={data_and_label['y_tr'].shape}")
        print(f"验证集形状: x_ts={data_and_label['x_ts'].shape}, y_ts={data_and_label['y_ts'].shape}")
        train_set = TensorDataset(
            torch.as_tensor(data_and_label["x_tr"], dtype=torch.float),
            torch.as_tensor(data_and_label["y_tr"], dtype=torch.long)
        )

        val_set = TensorDataset(
            torch.as_tensor(data_and_label["x_ts"], dtype=torch.float),
            torch.as_tensor(data_and_label["y_ts"], dtype=torch.long)
        )


        # 打印数据集大小
        logger.info(f"训练集大小: {len(train_set)}")
        logger.info(f"验证集大小: {len(val_set)}")

        # 修改DataLoader设置
        train_loader = DataLoader(
            train_set,
            batch_size=self.args.batch_size,  # 确保batch_size不超过数据集大小
            shuffle=True,
            drop_last=True  # 改为False以保留所有数据
        )

        val_loader = DataLoader(
            val_set,
            batch_size=self.args.batch_size,
            shuffle=False,
            drop_last=False
        )

        # 打印加载器信息
        logger.info(f"训练批次数: {len(train_loader)}")
        logger.info(f"验证批次数: {len(val_loader)}")

        with torch.no_grad():
            x_data = data_and_label["x_tr"]
            x_data = np.transpose(x_data, (1, 0, 2))
            c, n, f = x_data.shape
            train_x_for_cluster = x_data.reshape(c, n * f)
            cluster_labels, sub_sizes = auto_partition_channels(train_x_for_cluster, n_clusters=7, method='corr')
            print(f" cluster_labels =", cluster_labels)
            print(f" subgraph sizes =", sub_sizes)

        model = Mind(self.args, cluster_labels=cluster_labels)

        optimizer_model = optim.AdamW(
            model.parameters(),
            lr=self.args.lr1,
            weight_decay=self.args.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # 添加权重衰减
        _loss = CE_Label_Smooth_Loss(classes=self.args.n_class, epsilon=self.args.epsilon).to(self.args.device)
        model = model.to(self.args.device)

        train_epoch = self.args.epochs

        # 使用 ReduceLROnPlateau 学习率调度器
        lr_scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_model,
            mode='max',  # 如果监控准确率
            factor=0.5,  # 每次降低学习率的比例
            patience=20,  # 在多少个epoch后未改善时降低学习率
            verbose=True  # 输出调度器更新的日志
        )

        best_val_acc = 0
        best_f1 = 0
        best_codebook = []
        best_usage_rate_total = []

        # 添加模型保存目录创建
        os.makedirs(self.args.log_dir, exist_ok=True)
        center_loss_fn = CenterLoss(num_classes=self.args.n_class,
                            feat_dim=4140,
                            device=self.args.device,
                            lambda_c=0.9).to(self.args.device)
        optimizer_center = optim.SGD(center_loss_fn.parameters(), lr=0.5)

        for epoch in range(train_epoch):
            if len(train_loader) == 0:
                logger.error("❌ 训练数据加载器为空！请检查数据集和batch_size设置。")
                break

            progress_bar = tqdm(
                train_loader,
                desc=f'Train [{epoch + 1:03d}/{train_epoch}]',
                bar_format='{desc}: {percentage:3.0f}%|{bar:30}{r_bar}',
                dynamic_ncols=True,
                mininterval=1.0,
                ascii='->=',
                total=len(train_loader)  # 明确指定总数
            )

            usage1 = [[[0] * 32], [[0] * 16], [[0] * 16], [[0] * 16], [[0] * 16], [[0] * 16], [[0] * 16], [[0] * 16],
                      [[0] * 32]]

            model.train()
            train_acc = 0
            train_loss = 0
            total_train_samples = 0
            for i, (x, y) in enumerate(progress_bar):

                optimizer_model.zero_grad()
                optimizer_center.zero_grad()
                x, y = x.to(self.args.device), y.to(device=self.args.device, dtype=torch.int64)

                output, vq_loss, loss_aux, usage_tra, codebook_train, features = model(x)

                classification_loss = _loss(output, y) + 0.2 * vq_loss + 0.5 * loss_aux
                center_loss = center_loss_fn(features, y)
                loss = classification_loss + center_loss

                loss.backward()
                optimizer_model.step()
                optimizer_center.step()


                for usage_idx in range(9):
                    usage1[usage_idx].append(usage_tra[usage_idx])

                preds = torch.argmax(output, dim=1)
                correct = (preds == y).sum().item()
                batch_size = y.size(0)
                train_acc += correct
                train_loss += loss.item() * batch_size
                total_train_samples += batch_size

                # 更新进度条描述
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{train_acc / total_train_samples:.2%}'
                })

            train_acc /= total_train_samples
            train_loss /= total_train_samples

            usage2 = [[[0] * 32], [[0] * 16], [[0] * 16], [[0] * 16], [[0] * 16], [[0] * 16], [[0] * 16], [[0] * 16],
                      [[0] * 32]]

            val_progress = tqdm(
                val_loader,
                desc=f'val [{epoch + 1:03d}/{train_epoch}]',
                # 不用指定 total，让tqdm自动推断
                bar_format='{desc}: {percentage:3.0f}%|{bar:20}{r_bar}',
                dynamic_ncols=True,
                mininterval=1.0,
                ascii='->='
            )

            model.eval()
            total_val_samples = 0
            val_acc = 0
            val_loss = 0
            val_labels = []
            val_preds = []
            with torch.no_grad():
                for j, (a, b) in enumerate(val_progress):
                    a, b = a.to(self.args.device), b.to(device=self.args.device, dtype=torch.int64)
                    output, vq_loss_, aux_loss_, usage_val, codebook_eval, features_val = model(a)


                    preds = torch.argmax(output, dim=1)
                    correct = (preds == b).sum().item()
                    batch_size = b.size(0)
                    val_acc += correct
                    total_val_samples += batch_size
                    batch_loss = _loss(output, b) + 0.2 * vq_loss_ + 0.5 * aux_loss_
                    center_loss = center_loss_fn(features_val, b)
                    batch_loss = batch_loss + center_loss

                    val_loss += batch_loss.item() * batch_size
                    val_labels += b.cpu().numpy().tolist()
                    val_preds += preds.cpu().numpy().tolist()

                    correct_sofar = val_acc
                    total_sofar = (j + 1) * x.size(0)
                    val_progress.set_postfix({
                        'Loss': f'{batch_loss.item():.4f}',
                        'Acc': f'{correct_sofar/total_sofar:.2%}'
                    })
                lr_scheduler1.step(val_acc)

            usage_rate_train = []
            usage_rate_eval = []
            usage_rate_total = []
            for i in range(9):
                result1 = [sum(values) for values in zip(*usage1[i])]
                result2 = [sum(values) for values in zip(*usage2[i])]
                result3 = [x + y for x, y in zip(result1, result2)]
                usage_rate_train.append(result1)
                usage_rate_eval.append(result2)
                usage_rate_total.append(result3)

            val_acc = round(val_acc / total_val_samples, 4)
            val_loss = round(val_loss / total_val_samples, 4)
            f1_score = round(float(metrics.f1_score(val_labels, val_preds, labels=[0, 1, 2],
                                                    average='macro', zero_division=0)), 4)

            is_best_acc = 0
            is_best_f1 = 0
            if best_val_acc < val_acc:
                best_val_acc = val_acc
                best_f1 = f1_score
                best_codebook = codebook_eval
                best_usage_rate_train = usage_rate_train
                best_usage_rate_eval = usage_rate_eval
                best_usage_rate_total = usage_rate_total
                is_best_acc = 1

            if best_f1 < f1_score:
                best_f1 = f1_score

            if epoch == 0:
                logger.info(self.args)

            if epoch % 5 == 0:
                logger.info("val acc, f1 and loss on epoch_{} are: {}, {} and {}.".format(epoch, val_acc, f1_score,
                                                                                          val_loss))

            if epoch % 50 == 0:
                logger.info("now best val acc are: {}".format(best_val_acc))

            if best_val_acc == 0.99:
                break

            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(
                    self.args.log_dir,
                    f'checkpoint_{self.subject_name}_epoch{epoch + 1}.pth'
                )
                try:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': lr_scheduler1.state_dict(),
                        'train_acc': train_acc,
                        'val_acc': val_acc,
                        'val_loss': val_loss,
                        'f1_score': f1_score
                    }, checkpoint_path)
                    logger.info(f"💾 Epoch {epoch + 1}: 保存检查点到: {os.path.abspath(checkpoint_path)}")
                except Exception as e:
                    logger.error(f"❌ 保存检查点失败: {str(e)}")

            # 调整学习率

            # 记录每个epoch的指标
            logger.info(
                f"Epoch {epoch + 1}/{train_epoch} - "
                f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, "
                f"F1: {f1_score:.4f}, Loss: {val_loss:.4f}"
            )

            # 即使性能不好也保存最佳模型
            if val_acc >= best_val_acc:  # 改为>=，确保至少保存一次
                best_val_acc = val_acc
                best_f1 = f1_score
                best_codebook = codebook_eval
                best_usage_rate_total = usage_rate_total

                # 保存最佳模型
                best_model_path = os.path.join(
                    self.args.log_dir,
                    f'best_model_{self.subject_name}_acc{val_acc:.4f}_epoch{epoch + 1}.pth'
                )
                try:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': lr_scheduler1.state_dict(),
                        'best_val_acc': best_val_acc,
                        'best_f1': best_f1,
                        'train_acc': train_acc,
                        'val_loss': val_loss
                    }, best_model_path)
                    logger.info(f"✅ 保存最佳模型到: {os.path.abspath(best_model_path)}")
                except Exception as e:
                    logger.error(f"❌ 保存最佳模型失败: {str(e)}")

            # 如果验证准确率为1，提前停止
            if best_val_acc == 1:
                break
            print(best_val_acc)

        return best_val_acc, best_f1, best_codebook, best_usage_rate_total

    def save_checkpoint(self, model, epoch, is_best=False):
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': self.optimizer1.state_dict()
        }
        filename = os.path.join(self.args.log_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(state, filename)
        if is_best:
            best_filename = os.path.join(self.args.log_dir, 'model_best.pth')
            torch.save(state, best_filename)


def main():
    args = parse_args()
    print("当前设备:", args.device)
    if args.device == "cpu":
        print("警告：正在使用CPU运行，这可能会显著降低训练速度。")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    datatime_path = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    args.log_dir = os.path.join(args.log_dir, args.dataset, datatime_path)
    set_logging_config(args.log_dir)
    logger = logging.getLogger("main")
    logger.info("Logs and checkpoints will be saved to：{}".format(args.log_dir))
    logger.info(f"模型将保存至：{os.path.abspath(args.log_dir)}")

    print("当前工作目录:", os.getcwd())
    print("数据路径:", os.path.abspath(args.datapath))

    if not os.path.exists(args.datapath):
        raise FileNotFoundError(f"数据路径不存在: {args.datapath}")

    session_path = os.path.join(args.datapath, str(args.session))
    if not os.path.exists(session_path):
        raise FileNotFoundError(
            f"Session {args.session} 的数据路径不存在: {session_path}\n"
            f"请确保数据已正确放置在对应目录下。"
        )

    print(f"Session {args.session} 数据目录内容:", os.listdir(session_path))

    acc_list = []
    acc_dic = {}
    f1_list = []
    f1_dic = {}
    codebook_list = []
    codebook_dic = {}
    usage_dic = {}
    count = 0
    if args.dataset == 'SEED5' or args.dataset == 'MPED1':
        true_path = args.datapath
    else:
        true_path = os.path.join(args.datapath, str(args.session))

    if not os.path.exists(true_path):
        raise FileNotFoundError(
            f"数据路径不存在: {true_path}\n请确保数据文件已放置在正确位置，或通过--datapath参数指定正确的数据路径。")
    for subject in os.listdir(true_path):
        print(subject)
        if str(subject) == "label.mat":
            continue
        count += 1
        if args.dataset == 'SEED5':
            subject_name = str(subject).strip('.npz')
        else:
            subject_name = str(subject).strip('.npy')
        if args.mode == "dependent":

            logger.info(f"Dependent experiment on {count}th subject : {subject_name}")
            if args.dataset == 'SEED':
                data_and_label = load_data_denpendent1(true_path, subject)
            elif args.dataset == 'SEED4':
                data_and_label = load_data_denpendent(true_path, subject)
            elif args.dataset == 'SEED_new':
                data_and_label = load_data_denpendent1(true_path, subject)
            elif args.dataset == 'SEED4_new':
                data_and_label = load_data_denpendent2(true_path, subject)
            elif args.dataset == 'MPED':
                data_and_label = load_data_denpendent3(true_path, subject)
            else:
                data_and_label = load_data_de1(true_path, subject)
        elif args.mode == "independent":
            logger.info(f"Independent experiment on {count}th subject : {subject_name}")
            if args.dataset == 'SEED5':
                pass
            elif args.dataset == 'SEED4':
                data_and_label = load_data_inde1(true_path, subject)
            elif args.dataset == 'SEED4_new':
                data_and_label = load_data_inde1(true_path, subject)
            elif args.dataset == 'SEED':
                data_and_label = load_data_inde_yuan(true_path, subject)
            elif args.dataset == 'SEED_new':
                data_and_label = load_data_inde2(true_path, subject)
            elif args.dataset == 'MPED':
                data_and_label = load_data_inde3(true_path, subject)
            else:
                pass
        else:
            raise ValueError("Wrong mode selected.")

        trainer = Trainer(args, subject_name)
        valAcc, best_f1, best_codebook, best_usage = trainer.train(data_and_label)

        acc_list.append(valAcc)
        f1_list.append(best_f1)
        codebook_list.append(best_codebook)

        acc_dic[subject_name] = valAcc
        f1_dic[subject_name] = best_f1
        codebook_dic[subject_name] = best_codebook
        usage_dic[subject_name] = best_usage

        logger.info("Current best acc is : {}".format(acc_dic))
        logger.info("Current best f1 is : {}".format(f1_dic))
        logger.info("Current average acc is : {}, std is : {}".format(np.mean(acc_list), np.std(acc_list, ddof=1)))
        logger.info("Current average f1 is : {}, std is : {}".format(np.mean(f1_list), np.std(f1_list, ddof=1)))


def parse_args():
    parser = argparse.ArgumentParser()

    # 检测是否有可用的CUDA设备
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, default=device, help="gpu device")

    parser.add_argument("--log_dir", type=str, default="./logs", help="log file dir")
    parser.add_argument('--out_feature', type=int, default=20, help='Output feature for GCN.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay.')
    # hyperparameter
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to trai                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      n.')
    parser.add_argument('--lr1', type=float, default=0.001, help='Initial learning rate of SGD optimizer.')
    parser.add_argument('--lr2', type=float, default=0.005, help='Initial learning rate of Adam optimizer.')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size.')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate .')
    # pri-defined dataset
    parser.add_argument("--dataset", type=str, default="SEED4_new", help="dataset: SEED4, SEED5, MPED ,SEED,SEED4_new")
    parser.add_argument("--session", type=str, default="13", help="")
    parser.add_argument("--mode", type=str, default="dependent", help="dependent or independent")

    # 添加所有数据集相关的参数
    parser.add_argument("--in_feature", type=int, default=5, help="")
    parser.add_argument("--n_class", type=int, default=4, help="")
    parser.add_argument("--epsilon", type=float, default=0.01, help="")

    # 使用绝对路径
    default_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                "data", "SEED4_new")
    parser.add_argument("--datapath", type=str, default=default_path, help="数据路径")

    # 首先解析参数
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()
