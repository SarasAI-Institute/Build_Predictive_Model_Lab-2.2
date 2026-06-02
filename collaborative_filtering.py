# =============================================================================
# MODULE 2 | LAB 2.2
# File: 02_collaborative_filtering.py
# Purpose: Item-Item Collaborative Filtering on implicit purchase matrix
#          Evaluate Precision@10 vs Content-Based baseline from Lab 2.1
# Saras AI Institute | Build Predictive Models & Modern Recommenders
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  MODULE 2 | LAB 2.2")
print("  Item-Item Collaborative Filtering")
print("  Method: Implicit Purchase Matrix + Cosine Similarity")
print("=" * 60)

# ---------------------------------------------------------------------------
# SECTION 1: Load Events and Build Interaction Matrix
# ---------------------------------------------------------------------------
print("\n[1] Loading events and building implicit interaction matrix...")

events = pd.read_csv("data/events.csv")

# TODO: Define confidence weights for implicit user actions.
# Assign weights: 'view' -> 1, 'addtocart' -> 2, 'transaction' -> 3
event_weights = None

# TODO: Map your defined event weights dictionary onto the events['event'] column to populate a new column 'weight'
events['weight'] = None

print(f"    Total events    : {len(events):,}")


# TODO: Aggregate interaction strengths per unique user-item combination.
# Hint: Group the events dataframe by ['visitorid', 'itemid'], extract the 'weight' column, and call .sum(). Reset the index.
interactions = None

print(f"    Unique user-item pairs: {len(interactions) if interactions is not None else 0:,}")


# ---------------------------------------------------------------------------
# SECTION 2: Build Sparse Matrix
# ---------------------------------------------------------------------------
print("\n[2] Building sparse user-item matrix...")

# TODO: Extract lists of all unique user IDs ('visitorid') and item IDs ('itemid') from the interactions dataframe
user_ids  = None
item_ids  = None

# TODO: Generate continuous coordinate index maps (dictionaries) for users and items
# Format: {raw_id: coordinate_index_integer}
user_to_idx = None
item_to_idx = None

# TODO: Map the raw IDs inside your interactions dataframe to their respective structural coordinates
# Hint: Use .map() with your index dictionaries on 'visitorid' and 'itemid' columns to get underlying indices (.values)
row_idx = None
col_idx = None
data    = None

# TODO: Assemble a Compressed Sparse Row (CSR) matrix using your mapped coordinates and weights
# Hint: Pass a data configuration tuple ((data, (row_idx, col_idx))) alongside explicit shape constraints
user_item_matrix = None

# TODO: Transpose the user_item_matrix and format explicitly as a CSR matrix to facilitate item similarity calculations
item_user_matrix = None

print(f"    User-item matrix shape : {user_item_matrix.shape if user_item_matrix is not None else 'N/A'}")
print(f"    Non-zero entries       : {user_item_matrix.nnz:,}" if user_item_matrix is not None else "N/A")


# ---------------------------------------------------------------------------
# SECTION 3: Compute Item-Item Similarity
# ---------------------------------------------------------------------------
print("\n[3] Computing item-item similarity (batched for memory efficiency)...")

# TODO: Normalize item vectors using L2 norm criteria across the structural item_user_matrix profile
# Hint: Use sklearn's normalize() function specifying norm='l2'
item_user_norm = None

MAX_ITEMS = min(5000, len(item_ids)) if item_ids is not None else 5000
print(f"    Computing similarity for top {MAX_ITEMS:,} items (by interaction count)...")

# TODO: Identify the top most interacted items to isolate an executable benchmark evaluation boundary
# Hint: Compute horizontal sums across your item_user_matrix, flatten the array, use np.argsort()[::-1], and slice to MAX_ITEMS
top_item_indices = None
top_item_ids     = None

# TODO: Subset your normalized item matrix using top_item_indices and compute the baseline cosine similarity against all item vectors
# Hint: Call cosine_similarity(item_subset, item_user_norm)
item_subset = None
similarity_matrix = None


# ---------------------------------------------------------------------------
# SECTION 4: Build CF Recommendation Function
# ---------------------------------------------------------------------------
print("\n[4] Building CF recommendation function...")

def get_cf_recommendations(item_id, top_k=10):
    """
    Given an item_id, return top_k most similar items based on item-item collaborative filtering.
    Returns a pandas DataFrame tracking 'item_id' and 'similarity'.
    """
    # TODO: Check if item_id exists in item_to_idx mapping boundary. If not, return an empty DataFrame.
    if False:
        return pd.DataFrame()

    item_idx_in_catalog = item_to_idx[item_id]

    # TODO: Retrieve the full similarity score array for this item.
    # Logic: If item_id is inside top_item_ids, pull its precomputed vector row from similarity_matrix.
    # Otherwise, fall back to calculating its specific cosine similarity on the fly using item_user_norm.
    sims = None

    # TODO: Exclude the query item itself from recommendations by overriding its coordinate index in the sims array to -1
    
    # TODO: Discover indices tracking the top_k largest similarity values and build a return DataFrame matching target items
    top_indices = None

    return pd.DataFrame({
        'item_id'    : item_ids[top_indices] if item_ids is not None else [],
        'similarity' : sims[top_indices] if sims is not None else []
    })


# ---------------------------------------------------------------------------
# SECTION 5: Test the CF Recommender
# ---------------------------------------------------------------------------
print("\n[5] Testing CF recommender...")

if top_item_ids is not None:
    popular_item = top_item_ids[0]
    print(f"    Query item ID (most popular): {popular_item}")

    # TODO: Run your get_cf_recommendations routine to isolate matching candidate items for popular_item
    cf_recs = None
    print(cf_recs.to_string(index=False) if cf_recs is not None else "    Not Implemented")


# ---------------------------------------------------------------------------
# SECTION 6: Evaluate Precision@10 vs CB Baseline
# ---------------------------------------------------------------------------
print("\n[6] Evaluating CF — Precision@10 vs CB baseline...")

# Load Content-Based results from Lab 2.1 archive artifact
try:
    with open("data/cb_artifacts.pkl", "rb") as f:
        cb_artifacts = pickle.load(f)
    cb_precision = cb_artifacts['cb_results']['precision_at_10']
    print(f"    CB Precision@10 (Lab 2.1): {cb_precision:.4f}")
except:
    cb_precision = None
    print("    CB artifacts not found — run Lab 2.1 first")

# TODO: Isolate transactional events to verify retrieval scores across users with multiple purchases
# Filter events down to 'transaction' values and subset column structures to ['visitorid', 'itemid']
purchases = None
if purchases is not None:
    purchases.columns = ['user_id', 'item_id']

# TODO: Isolate sequential multi-purchase records by keeping users with 2 or more historical purchases
multi_buyers = None

cf_hits   = 0
cf_total  = 0

if multi_buyers is not None:
    eval_users = multi_buyers['user_id'].unique()[:500]

    for user in eval_users:
        user_items    = purchases[purchases['user_id'] == user]['item_id'].values
        catalog_items = [i for i in user_items if i in item_to_idx]

        if len(catalog_items) < 2:
            continue

        query_item  = catalog_items[0]
        target_item = catalog_items[1]

        # TODO: Call get_cf_recommendations using query_item to extract predicted recommendations
        recs = None
        if recs is None or len(recs) == 0:
            continue

        # TODO: Check if target_item exists inside predicted item vectors. 
        # If true, increment cf_hits. Accumulate cf_total count per user track evaluated.
        pass

# TODO: Compute ultimate cf_precision ratio tracking performance across valid boundaries
cf_precision = 0.0

print(f"\n    Evaluated on {cf_total} users")
print(f"    CF Hits@10     : {cf_hits}")
print(f"    CF Precision@10: {cf_precision:.4f}")


# ---------------------------------------------------------------------------
# SECTION 7: Sparsity Problem Visualization
# ---------------------------------------------------------------------------
print("\n[7] Visualizing the sparsity problem...")

# --- Rendering Long Tail Distributions ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Lab 2.2: Collaborative Filtering — The Sparsity Problem", fontsize=13, fontweight='bold')

# TODO: Plot an item interaction frequency distribution histogram using axes[0] on a logarithmic scale
# Hint: Use axes[0].hist() and configure axes[0].set_yscale('log') or set log=True


axes[0].set_title("Item Interaction Count Distribution\n(log scale — power law)")
axes[0].set_xlabel("Number of Interactions per Item")
axes[0].set_ylabel("Number of Items (log)")

# TODO: Plot user interaction frequency distribution counts using a log histogram format on axes[1]


axes[1].set_title("User Interaction Count Distribution\n(log scale — power law)")
axes[1].set_xlabel("Number of Interactions per User")
axes[1].set_ylabel("Number of Users (log)")

plt.tight_layout()
plt.savefig("output/02_cf_sparsity_analysis.png", dpi=150, bbox_inches='tight')
plt.show()


# ---------------------------------------------------------------------------
# SECTION 8: Save CF Artifacts for Lab 2.4
# ---------------------------------------------------------------------------
# TODO: Dump your sparse matrix definitions, catalog arrays, index mappings, and calculated metrics to a pickle file
# Path: "data/cf_artifacts.pkl"


print("\n    Saved -> data/cf_artifacts.pkl")
print("    Move to: 03_lightfm_cold_start.py")
