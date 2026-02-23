import json
import os
import torch
import torch.optim as optim
from gdown import download


def download_data(id, filename):
    if not os.path.exists(filename):
        download(id=id, output=filename, quiet=False)
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

movies = download_data('1bwVnwW9FL4-pMUpOTzXOomHc150NncmW', 'movies.json')
levels = download_data('1O_vJ7aoWfakDKFCZb0zc51BjzUoLJrCW', 'levels.json')
embeddings = torch.tensor(download_data('1E1lHQQ09yQWSw-YWmsnRt8ukbV7-Gznr', 'embeddings.json'))


def find_optimal_embedding(E, target_order=None, n_restarts=10, n_steps=4000, lr=0.005, verbose=True):
    """
    Βρίσκει embedding q που ταξινομεί τις target ταινίες πρώτες με μέγιστα gaps.

    E             : [n_movies, 384] tensor, ήδη normalized
    target_order  : [5] tensor των indices που θέλουμε πρώτους (default [0,1,2,3,4])
    n_restarts    : αριθμός επανεκκινήσεων με διαφορετικό noise
    n_steps       : βήματα gradient descent ανά restart
    lr            : learning rate
    """
    if target_order is None:
        target_order = torch.tensor([0, 1, 2, 3, 4])

    n, d = E.shape
    # Εφόσον τα embeddings είναι ήδη normalized, η cosine similarity = dot product
    E_norm = E / E.norm(dim=-1, keepdim=True)

    # Mask για non-target ταινίες
    non_target_mask = torch.ones(n, dtype=torch.bool)
    non_target_mask[target_order] = False

    best_q = None
    best_min_gap = -float('inf')
    best_correct = False

    for restart in range(n_restarts):
        # Αρχικοποίηση: weighted average των target embeddings + noise
        # Φθίνοντα βάρη: ταινία 0 → 5x, ταινία 4 → 1x
        w = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
        q_init = (w.unsqueeze(1) * E_norm[target_order]).sum(0)
        q_init = q_init / q_init.norm()

        noise_scale = 0.1 * restart  # restart 0: καθαρή αρχικοποίηση
        if restart > 0:
            q_init = q_init + noise_scale * torch.randn(d)

        q = q_init.detach().clone().requires_grad_(True)
        optimizer = optim.Adam([q], lr=lr)

        for step in range(n_steps):
            optimizer.zero_grad()

            q_norm = q / q.norm()
            sim = E_norm @ q_norm  # cosine similarities για όλες τις ταινίες

            t_sim = sim[target_order]          # [5] similarities των targets (θέλουμε φθίνουσες)
            o_max = sim[non_target_mask].max() # max similarity των non-targets

            # 5 gaps που θέλουμε να μεγιστοποιήσουμε (όλα > 0 για σωστή σειρά)
            gaps = torch.cat([
                t_sim[:-1] - t_sim[1:],               # 4 gaps μεταξύ consecutive targets
                (t_sim[-1] - o_max).unsqueeze(0)       # gap: target[4] vs καλύτερος non-target
            ])

            # Soft-min (differentiable προσέγγιση του min) — αυξάνουμε temperature σταδιακά
            temp = min(10.0 + step * 0.02, 80.0)
            soft_min = -torch.logsumexp(-temp * gaps, dim=0) / temp

            # Penalty για παραβίαση σειράς (κρατάει το optimization σωστό)
            penalty = 100.0 * torch.clamp(-gaps, min=0).sum()

            loss = -soft_min + penalty
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            q_eval = (q / q.norm()).clone()
            sim = E_norm @ q_eval
            ranking = sim.argsort(descending=True)
            correct = torch.all(ranking[:5] == target_order)

            t_sim = sim[target_order]
            o_max = sim[non_target_mask].max()
            gaps = torch.cat([
                t_sim[:-1] - t_sim[1:],
                (t_sim[-1] - o_max).unsqueeze(0)
            ])
            min_gap = gaps.min().item()

        if verbose:
            status = "✓" if correct.item() else "✗"
            print(f"  [{status}] restart {restart+1:2d}: correct={correct.item()}, "
                  f"min_gap={min_gap:.5f}, top5={ranking[:5].tolist()}")

        # Κρατάμε το καλύτερο (προτεραιότητα: σωστή σειρά, μετά μεγαλύτερο gap)
        is_better = (
            (correct.item() and not best_correct) or
            (correct.item() == best_correct and min_gap > best_min_gap)
        )
        if is_better:
            best_correct = correct.item()
            best_min_gap = min_gap
            best_q = q_eval

    return best_q


answers = {}

for levelid in ['level1', 'level2', 'level3', 'level4']:
    level = levels[levelid]
    target_order = torch.tensor([0, 1, 2, 3, 4])
    E = embeddings[level]

    print(f"\n{'='*55}")
    print(f"  {levelid}  ({len(level)} ταινίες)")
    for i, idx in enumerate(level[:5]):
        print(f"  [{i}] {movies[idx]['title']}")
    print(f"{'='*55}")

    best_emb = find_optimal_embedding(E, target_order, n_restarts=10, n_steps=4000)

    E_norm = E / E.norm(dim=-1, keepdim=True)
    sim = E_norm @ best_emb
    ranking = sim.argsort(descending=True)
    diffs = -sim[ranking].diff()[:5]

    print(f"\n  >> Αποτέλεσμα:")
    print(f"     ranking[:6] = {ranking[:6].tolist()}")
    print(f"     Σωστή σειρά = {torch.all(ranking[:5] == target_order).item()}")
    print(f"     Diffs: {[f'{d:.5f}' for d in diffs.tolist()]}")
    print(f"     Min diff: {diffs.min():.5f}")

    answers[levelid] = {'embedding': best_emb.tolist()}

with open('answers.json', 'w') as f:
    json.dump(answers, f)
print("Αποθηκεύτηκε στο answers.json")
