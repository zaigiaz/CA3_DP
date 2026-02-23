# Shayne Tieman 
# CS 490/590 AA
# Coding Assignment #3
# Extra Credit | Not Attempted

from collections import deque, defaultdict
import timeit
from memory_profiler import profile
from itertools import combinations
from itertools import permutations
import random
import sys

# find all possible sums and mark true in dp table
def subset_sum_dp(values):
    S = sum(values)

    # dp table of whether we can reach sum (true, false)
    possible = [False] * (S + 1)

    # table of item index and prev_sum used to reach it
    parent = [None] * (S + 1)

    possible[0] = True

    # o(n^2) double for loop
    # tabulation because we iterate backwards and track when adding v to previous sum s is possible
    for i, v in enumerate(values):
        for s in range(S, v - 1, -1):
            if not possible[s] and possible[s - v]:
                possible[s] = True

                # parent[s] = (item_index, prev_sum) used to reach s
                parent[s] = (i, s - v)

    return possible, parent


# recover indices forming target using parent table. return list of indices
def recover_subset(parent, target):

    res = []
    cur = target

    while cur and parent[cur] is not None:
        i, prev = parent[cur]
        res.append(i)
        cur = prev

    if cur != 0:
        raise ValueError("Target not reachable")

    res.reverse()
    return res


# allocate cards if one favorite is chosen
# TODO: check to see correct when no favorite is chosen, then everything zero
def allocate_for_G1(values, selected, favorite):
    n = len(values)

    result = {0: ([],0), 1: ([],0), 2: ([],0)}
    discarded = []

    if favorite in selected:
        result[favorite] = (list(range(n)), sum(values))
    else:
        discarded = list(range(n))
    return result, discarded


# allocate cards if two favorites are chosen
def allocate_for_G2(values, selected, favorite):
    n = len(values)
    all_gkids = [0,1,2]
    result = {g: ([],0) for g in all_gkids}
    discarded = []

    left_g, right_g = selected[0], selected[1]
    fav_included = favorite in selected

    A_idx, B_idx, sumA, sumB = two_way_partition(values, prefer_larger_for_index=None)

    # if ther is a favorite then they have the larger sum out of two sets
    if fav_included:

        if sumA >= sumB:
            if left_g == favorite:
                result[left_g] = (A_idx, sumA) 
                result[right_g] = (B_idx, sumB)
            else:
                result[right_g] = (A_idx, sumA)
                result[left_g] = (B_idx, sumB)
        else:
            if left_g == favorite:
                result[left_g] = (B_idx, sumB)
                result[right_g] = (A_idx, sumA)
            else:
                result[right_g] = (B_idx, sumB)
                result[left_g] = (A_idx, sumA)
    else:
        # favorite excluded: require equal halves, discard minimally if needed
        discard, remaining = find_min_discard_for_equal_partition(values, G=2)
        if discard is None:
            result[left_g] = (A_idx, sumA)
            result[right_g] = (B_idx, sumB)

        else:
            discarded = discard
            rem_vals = [values[i] for i in remaining]
            poss, par = subset_sum_dp(rem_vals)
            target = sum(rem_vals) // 2
            sub_local = recover_subset(par, target)

            A_idx = [remaining[i] for i in sub_local]
            B_idx = [i for i in remaining if i not in A_idx]
            result[left_g] = (A_idx, sum(values[i] for i in A_idx))
            result[right_g] = (B_idx, sum(values[i] for i in B_idx))

    return result, discarded


# allocate cards if 3 favorites are chosen
def allocate_for_G3(values, selected_indices, favorite_index, second_fav_index=None):
    n = len(values)
    result = {0: ([], 0), 1: ([], 0), 2: ([], 0)}
    discarded = []

    fav_included = favorite_index in selected_indices
    
    if not fav_included:
        discard, remaining = find_min_discard_for_equal_partition(values, G=3)
        if discard is None:
            remaining = list(range(len(values)))
        else:
            discarded = discard
    else:
        remaining = list(range(len(values)))

    rem_vals = [values[i] for i in remaining]
    
    if rem_vals:
        A_local, rest_local, sumA, sumRest = two_way_partition(rem_vals, prefer_larger_for_index=None)
        A_idx = [remaining[i] for i in A_local]
        rest_idx = [remaining[i] for i in rest_local]

        rest_vals = [values[i] for i in rest_idx]
        
        if rest_vals:
            B_local, C_local, sumB, sumC = two_way_partition(rest_vals, prefer_larger_for_index=None)
            B_idx = [rest_idx[i] for i in B_local]
            C_idx = [rest_idx[i] for i in C_local]
        else:
            B_idx = []
            C_idx = []
            sumB = 0
            sumC = 0

        buckets = [(A_idx, sumA), (B_idx, sumB), (C_idx, sumC)]

        # Try permutations to satisfy fav and second_fav constraints
        assigned = None
        
        for perm in permutations(buckets, len(selected_indices)):
            ok = True
            
            perm_sums = [perm[i][1] for i in range(len(perm))]
            max_sum = max(perm_sums)
            sorted_sums = sorted(perm_sums, reverse=True)
            
            # Check favorite constraint: must be in largest group
            if fav_included:
                fav_idx_in_sel = selected_indices.index(favorite_index)
                fav_sum = perm[fav_idx_in_sel][1]
                if fav_sum < max_sum:
                    ok = False
            
            # Check second_fav constraint: must be in second-largest group
            if ok and second_fav_index is not None and second_fav_index in selected_indices:
                second_fav_idx_in_sel = selected_indices.index(second_fav_index)
                second_fav_sum = perm[second_fav_idx_in_sel][1]
                
                # second_fav_sum should equal second largest sum
                if len(sorted_sums) >= 2:
                    second_largest = sorted_sums[1]
                    if second_fav_sum != second_largest:
                        ok = False
                else:
                    ok = False
            
            if ok:
                assigned = {selected_indices[i]: perm[i] for i in range(len(selected_indices))}
                break
        
        if assigned is None:
            assigned = {selected_indices[i]: buckets[i] for i in range(len(selected_indices))}

        for g in selected_indices:
            idxs, s = assigned.get(g, ([], 0))
            result[g] = (idxs, s)

    return result, discarded


# find the min discard so that the other two grandchildren can have equal subset-sum of cards
# @profile
def find_min_discard_for_equal_partition(values, G):
    S = sum(values)
    # target remaining sum must be divisible by G
    possible_discard, parent_discard = subset_sum_dp(values)
    candidates = []

    for d in range(0, S + 1):
        if not possible_discard[d]:
            continue
        if (S - d) % G != 0:
            continue
        candidates.append(d)

    for d in candidates:
        discard_indices = set(recover_subset(parent_discard, d))
        rem_values = [values[i] for i in range(len(values)) if i not in discard_indices]
        # try to partition remaining into G equal parts
        target = (S - d) // G
        feasible = True
        assigned = []
        rem_indices = [i for i in range(len(values)) if i not in discard_indices]
        # extract subsets equal to target using DP on the remaining list
        rem_vals = rem_values[:]
        rem_inds = rem_indices[:]

        for _ in range(G - 1):
            poss, par = subset_sum_dp(rem_vals)
            if not (0 <= target <= sum(rem_vals) and poss[target]):
                feasible = False
                break
            subset_local = recover_subset(par, target)
            # map local indices to original indices
            chosen_orig = [rem_inds[idx] for idx in subset_local]
            assigned.append(chosen_orig)
            # remove chosen from rem_vals/rem_inds
            mask = set(subset_local)
            rem_vals = [v for idx, v in enumerate(rem_vals) if idx not in mask]
            rem_inds = [idx for idx in rem_inds if idx not in mask]
        if not feasible:
            continue
        # last bucket is remaining rem_inds
        if sum(values[i] for i in rem_inds) != target:
            feasible = False
        if feasible:
            discard_list = sorted(discard_indices)
            remaining_list = sorted(i for i in range(len(values)) if i not in discard_indices)
            return discard_list, remaining_list
    return None, None


# Two way partition of set trying to minimize the diff between two subset-sums
# @profile
def two_way_partition(values, prefer_larger_for_index=None):
    S = sum(values)
    possible, parent = subset_sum_dp(values)

    best = None  
    for t in range(0, S + 1):
        if not possible[t]:
            continue

        diff = abs(2 * t - S)
        
        # find best t (sum in dp table) that minimize constraint abs(2t - S)
        if best is None or diff < best[0] or (diff == best[0] and t > best[1]):
            best = (diff, t)

    # if no value satisfy return all values (cant partition)
    if best is None:
        return [], list(range(len(values))), 0, S

    # If favorite constraint, ensure the subset containing that index is the larger of two
    if prefer_larger_for_index is not None:
        best_candidates = []
        min_diff = best[0]

        for t in range(0, S + 1):
            if not possible[t]:
                continue
            if abs(2 * t - S) != min_diff:
                continue

            A = set(recover_subset(parent, t))
            larger_is_A = (t >= S - t)
            contains_fav = prefer_larger_for_index in A
        
            if (larger_is_A and contains_fav) or (not larger_is_A and not contains_fav):
                best_candidates.append((t, A))

        if best_candidates:
            t, A = best_candidates[0]
        else:
            chosen = None
            for t in range(0, S + 1):

                if not possible[t]:
                    continue

                diff = abs(2 * t - S)
                A = set(recover_subset(parent, t))
                larger_is_A = (t >= S - t)
                contains_fav = prefer_larger_for_index in A

                if (larger_is_A and contains_fav) or (not larger_is_A and not contains_fav):
                    if chosen is None or diff < chosen[0] or (diff == chosen[0] and t > chosen[1]):
                        chosen = (diff, t)

            if chosen is not None:
                t = chosen[1]
                A = set(recover_subset(parent, t))
            else:
                # fallback to previously found best
                t = best[1]
                A = set(recover_subset(parent, t))
    else:
        t = best[1]
        A = set(recover_subset(parent, t))

    A_indices = sorted(A)
    B_indices = [i for i in range(len(values)) if i not in A_indices]
    sumA = sum(values[i] for i in A_indices)
    sumB = sum(values[i] for i in B_indices)
    return A_indices, B_indices, sumA, sumB


def time_complexity(A, G, name):
    T = 1
       
    t = timeit.Timer(lambda: ST_ca3(A, G, name)) 
    print("time-complexity (seconds): ", t.timeit(T))
 


# use random.choices() to generate card values from 1-50 N times
# where N is size of array
def gen_card_values(N):
    l = random.choices(list(range(1, 51)), k=N)
    return l

@profile
def ST_ca3(A, G, name):

    # TODO: refactor interface to get indices give the name fo the function

    names_key = {"Camila":0,"Melanie":1,"Selena":2}
    names_value = {0:"Camila",1:"Melanie",2:"Selena"}
    new_name = names_key.get(name)
    values = A

    # select amount of grandkids to use
    selected = random.sample([0,1,2], G)
    second_fav = None

    # use each allocation function depending on how many favorites grandma will have
    # there needs to be second fav if there is 3 grandkids 
    if G == 3:
        second_fav = random.choice([g for g in [0,1,2] if g != new_name])
        alloc, discarded = allocate_for_G3(values, selected, new_name, second_fav)
        
    if G == 2:
        alloc, discarded = allocate_for_G2(values, selected, new_name)
        
    # one grandkid means we default to giving favorite total-sum or nothing to anyone
    if G == 1:
        alloc, discarded = allocate_for_G1(values, selected, new_name)

    print("Values:", values)
    print("G:", G, "Selected:", selected, "Favorite:", name, end=' ')

    if second_fav != None:
        print("Second fav:", names[second_fav])

    else:
        print("Second fav: None")

    for g in [0,1,2]:
       idxs, s = alloc[g]
       print(f"{names_value.get(g)} -> cards: {idxs}, total: {s}")
    
    print("Discarded indices:", discarded)
    


G = random.choice([1,2,3])
A = gen_card_values(1024)

# favorite grandkid based on initials = 1 ((18 + 19) % 3 == 1)
ST_grandkid = 1

names = {0:"Camila",1:"Melanie",2:"Selena"}
choice = random.choice([0,1,2])
name = names[choice]

ST_ca3(A, G, name)

# print("\nhere is testing variables with timeit functions:")
# time_complexity(A, G, name)
