import time
import torch
from tqdm import tqdm
from helpers import list_of_distances, make_one_hot

device = torch.device("cuda" if torch.cuda.is_available() 
                      else ("mps" if torch.backends.mps.is_available() else "cpu"))

def get_base_model(model):
    return model.module if hasattr(model, 'module') else model

def _train_or_test(model, dataloader, optimizer=None, class_specific=True, use_l1_mask=True,
                   coefs=None, log=print):
    is_train = optimizer is not None
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    total_cluster_cost = 0
    total_separation_cost = 0
    total_avg_separation_cost = 0

    base_model = get_base_model(model)

    # Use tqdm to display progress across batches
    for i, (image, label) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Batches"):
        input = image.to(device)
        target = label.to(device)

        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            output, min_distances = model(input)
            cross_entropy = torch.nn.functional.cross_entropy(output, target)
            if class_specific:
                max_dist = (base_model.prototype_shape[1] *
                            base_model.prototype_shape[2] *
                            base_model.prototype_shape[3])
                prototypes_of_correct_class = torch.t(base_model.prototype_class_identity[:, label]).to(device)
                inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
                cluster_cost = torch.mean(max_dist - inverted_distances)
                prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                inverted_distances_to_nontarget_prototypes, _ = torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
                separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)
                avg_separation_cost = torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)
                avg_separation_cost = torch.mean(avg_separation_cost)
                
                if use_l1_mask:
                    l1_mask = 1 - torch.t(base_model.prototype_class_identity).to(device)
                    l1 = (base_model.last_layer.weight * l1_mask).norm(p=1)
                else:
                    l1 = base_model.last_layer.weight.norm(p=1) 
            else:
                min_distance, _ = torch.min(min_distances, dim=1)
                cluster_cost = torch.mean(min_distance)
                l1 = base_model.last_layer.weight.norm(p=1)

            _, predicted = torch.max(output.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()
            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_cluster_cost += cluster_cost.item()
            total_separation_cost += separation_cost.item()
            total_avg_separation_cost += avg_separation_cost.item()

        if is_train:
            if class_specific:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy +
                            coefs['clst'] * cluster_cost +
                            coefs['sep'] * separation_cost +
                            coefs['l1'] * l1)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1
            else:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy +
                            coefs['clst'] * cluster_cost +
                            coefs['l1'] * l1)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost + 1e-4 * l1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        del input, target, output, predicted, min_distances

    end = time.time()
    log('\ttime: \t{0}'.format(end - start))
    log('\tcross ent: \t{0}'.format(total_cross_entropy / n_batches))
    log('\tcluster: \t{0}'.format(total_cluster_cost / n_batches))
    if class_specific:
        log('\tseparation:\t{0}'.format(total_separation_cost / n_batches))
        log('\tavg separation:\t{0}'.format(total_avg_separation_cost / n_batches))
    log('\taccu: \t\t{0}%'.format(n_correct / n_examples * 100))
    log('\tl1: \t\t{0}'.format(base_model.last_layer.weight.norm(p=1).item()))
    p = base_model.prototype_vectors.view(base_model.num_prototypes, -1).cpu()
    with torch.no_grad():
        p_avg_pair_dist = torch.mean(list_of_distances(p, p))
    log('\tp dist pair: \t{0}'.format(p_avg_pair_dist.item()))

    return n_correct / n_examples

def train(model, dataloader, optimizer, class_specific=False, coefs=None, log=print):
    assert(optimizer is not None)
    log('\ttrain')
    model.train()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=optimizer,
                          class_specific=class_specific, coefs=coefs, log=log)


def test(model, dataloader, class_specific=False, log=print):
    log('\ttest')
    model.eval()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=None,
                          class_specific=class_specific, log=log)


def last_only(model, log=print):
    base_model = get_base_model(model)
    for p in base_model.features.parameters():
        p.requires_grad = False
    for p in base_model.add_on_layers.parameters():
        p.requires_grad = False
    base_model.prototype_vectors.requires_grad = False
    for p in base_model.last_layer.parameters():
        p.requires_grad = True
    
    log('\tlast layer')


def warm_only(model, log=print):
    base_model = get_base_model(model)
    for p in base_model.features.parameters():
        p.requires_grad = False
    for p in base_model.add_on_layers.parameters():
        p.requires_grad = True
    base_model.prototype_vectors.requires_grad = True
    for p in base_model.last_layer.parameters():
        p.requires_grad = True
    
    log('\twarm')


def joint(model, log=print):
    base_model = get_base_model(model)
    for p in base_model.features.parameters():
        p.requires_grad = True
    for p in base_model.add_on_layers.parameters():
        p.requires_grad = True
    base_model.prototype_vectors.requires_grad = True
    for p in base_model.last_layer.parameters():
        p.requires_grad = True
    
    log('\tjoint')