import copy

import numpy as np
import torch


class Memory:
    def __init__(self, memory_size, nb_total_classes, rehearsal, fixed=True, modes=1):
        self.memory_size = memory_size
        self.nb_total_classes = nb_total_classes
        self.rehearsal = rehearsal
        self.fixed = fixed
        self.modes = modes

        self.x = self.y = self.t = None

        self.nb_classes = 0

    @property
    def memory_per_class(self):
        if self.fixed:
            return self.memory_size // self.nb_total_classes
        return self.memory_size // self.nb_classes if self.nb_classes > 0 else self.memory_size

    def get_dataset(self, base_dataset):
        dataset = copy.deepcopy(base_dataset)
        dataset._x = self.x
        dataset._y = self.y
        dataset._t = self.t

        return dataset

    def get(self):
        return self.x, self.y, self.t

    def __len__(self):
        return len(self.x) if self.x is not None else 0

    def save(self, path):
        np.savez(
            path,
            x=self.x, y=self.y, t=self.t
        )

    def load(self, path):
        data = np.load(path)
        self.x = data["x"]
        self.y = data["y"]
        self.t = data["t"]

        assert len(self) <= self.memory_size, len(self)
        self.nb_classes = len(np.unique(self.y))

    def reduce(self):
        x, y, t = [], [], []
        for class_id in np.unique(self.y):
            indexes = np.where(self.y == class_id)[0]
            if self.modes > 1:
                selected_indexes = np.concatenate([
                    indexes[:len(indexes)//2][:self.memory_per_class//2],
                    indexes[len(indexes)//2:][:self.memory_per_class//2],
                ])
            else:
                selected_indexes = indexes[:self.memory_per_class]
            x.append(self.x[selected_indexes])
            y.append(self.y[selected_indexes])
            t.append(self.t[selected_indexes])

        self.x = np.concatenate(x)
        self.y = np.concatenate(y)
        self.t = np.concatenate(t)

    def add(self, dataset, model, nb_new_classes):
        self.nb_classes += nb_new_classes

        if self.modes > 1:  # todo modes more than 2
            assert self.modes == 2
            x1, y1, t1 = herd_samples(dataset, model, self.memory_per_class//2, self.rehearsal)
            x2, y2, t2 = herd_samples(dataset, model, self.memory_per_class//2, self.rehearsal)
            x = np.concatenate((x1, x2))
            y = np.concatenate((y1, y2))
            t = np.concatenate((t1, t2))
        else:
            x, y, t = herd_samples(dataset, model, self.memory_per_class, self.rehearsal)
        #assert len(y) == self.memory_per_class * nb_new_classes, (len(y), self.memory_per_class, nb_new_classes)

        if self.x is None:
            self.x, self.y, self.t = x, y, t
        else:
            if not self.fixed:
                self.reduce()
            self.x = np.concatenate((self.x, x))
            self.y = np.concatenate((self.y, y))
            self.t = np.concatenate((self.t, t))


def herd_samples(dataset, model, memory_per_class, rehearsal):
    x, y, t = dataset._x, dataset._y, dataset._t

    if rehearsal == "random":
        indexes = []
        for class_id in np.unique(y):
            class_indexes = np.where(y == class_id)[0]
            indexes.append(
                np.random.choice(class_indexes, size=memory_per_class)
            )
        indexes = np.concatenate(indexes)

        return x[indexes], y[indexes], t[indexes]
    elif "closest" in rehearsal:
        if rehearsal == 'closest_token':
            handling = 'last'
        else:
            handling = 'all'

        features, targets = extract_features(dataset, model, handling)
        indexes = []

        for class_id in np.unique(y):
            class_indexes = np.where(y == class_id)[0]
            class_features = features[class_indexes]

            class_mean = np.mean(class_features, axis=0, keepdims=True)
            distances = np.power(class_features - class_mean, 2).sum(-1)
            class_closest_indexes = np.argsort(distances)

            indexes.append(
                class_indexes[class_closest_indexes[:memory_per_class]]
            )

        indexes = np.concatenate(indexes)
        return x[indexes], y[indexes], t[indexes]
    elif "furthest" in rehearsal:
        if rehearsal == 'furthest_token':
            handling = 'last'
        else:
            handling = 'all'

        features, targets = extract_features(dataset, model, handling)
        indexes = []

        for class_id in np.unique(y):
            class_indexes = np.where(y == class_id)[0]
            class_features = features[class_indexes]

            class_mean = np.mean(class_features, axis=0, keepdims=True)
            distances = np.power(class_features - class_mean, 2).sum(-1)
            class_furthest_indexes = np.argsort(distances)[::-1]

            indexes.append(
                class_indexes[class_furthest_indexes[:memory_per_class]]
            )

        indexes = np.concatenate(indexes)
        return x[indexes], y[indexes], t[indexes]
    elif "icarl":
        if rehearsal == 'icarl_token':
            handling = 'last'
        else:
            handling = 'all'

        features, targets = extract_features(dataset, model, handling)
        indexes = []

        for class_id in np.unique(y):
            class_indexes = np.where(y == class_id)[0]
            class_features = features[class_indexes]

            indexes.append(
                class_indexes[icarl_selection(class_features, memory_per_class)]
            )

        indexes = np.concatenate(indexes)
        return x[indexes], y[indexes], t[indexes]
    else:
        raise ValueError(f"Unknown rehearsal method {rehearsal}!")



def extract_features(dataset, model, ensemble_handling='last'):
    #transform = copy.deepcopy(dataset.trsf.transforms)
    #dataset.trsf = transforms.Compose(transform[-2:])

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
        shuffle=False
    )

    features, targets = [], []

    with torch.no_grad():
        for x, y, _ in loader:
            if hasattr(model, 'module'):
                feats, _, _ = model.module.forward_features(x.cuda())
            else:
                feats, _, _ = model.forward_features(x.cuda())

            if isinstance(feats, list):
                if ensemble_handling == 'last':
                    feats = feats[-1]
                elif ensemble_handling == 'all':
                    feats = torch.cat(feats, dim=1)
                else:
                    raise NotImplementedError(f'Unknown handdling of multiple features {ensemble_handling}')
            elif len(feats.shape) == 3:  # joint tokens
                if ensemble_handling == 'last':
                    feats = feats[-1]
                elif ensemble_handling == 'all':
                    feats = feats.permute(1, 0, 2).view(len(x), -1)
                else:
                    raise NotImplementedError(f'Unknown handdling of multiple features {ensemble_handling}')

            feats = feats.cpu().numpy()
            y = y.numpy()

            features.append(feats)
            targets.append(y)

    features = np.concatenate(features)
    targets = np.concatenate(targets)

    #dataset.trsf = transforms.Compose(transform)
    return features, targets


def icarl_selection(features, nb_examplars):
    D = features.T
    D = D / (np.linalg.norm(D, axis=0) + 1e-8)
    mu = np.mean(D, axis=1)
    herding_matrix = np.zeros((features.shape[0],))

    w_t = mu
    iter_herding, iter_herding_eff = 0, 0

    while not (
        np.sum(herding_matrix != 0) == min(nb_examplars, features.shape[0])
    ) and iter_herding_eff < 1000:
        tmp_t = np.dot(w_t, D)
        ind_max = np.argmax(tmp_t)
        iter_herding_eff += 1
        if herding_matrix[ind_max] == 0:
            herding_matrix[ind_max] = 1 + iter_herding
            iter_herding += 1

        w_t = w_t + mu - D[:, ind_max]

    herding_matrix[np.where(herding_matrix == 0)[0]] = 10000

    return herding_matrix.argsort()[:nb_examplars]


def get_finetuning_dataset(dataset, memory, finetuning='balanced', oversample_old=1, task_id=0):
    if finetuning == 'balanced':
        x, y, t = memory.get()

        if oversample_old > 1:
            old_indexes = np.where(t < task_id)[0]
            assert len(old_indexes) > 0
            new_indexes = np.where(t >= task_id)[0]

            indexes = np.concatenate([
                np.repeat(old_indexes, oversample_old),
                new_indexes
            ])
            x, y, t = x[indexes], y[indexes], t[indexes]

        new_dataset = copy.deepcopy(dataset)
        new_dataset._x = x
        new_dataset._y = y
        new_dataset._t = t
    elif finetuning in ('all', 'none'):
        new_dataset = dataset
    else:
        raise NotImplementedError(f'Unknown finetuning method {finetuning}')

    return new_dataset
