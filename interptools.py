import pandas as pd
import numpy as np 
from sklearn.tree import DecisionTreeClassifier

def get_rules(dtc, df, master_df):
    
    rules_list = []
    values_path = []
    leaf_list = []
    values = dtc.tree_.value
    
    X = df.copy()

    def RevTraverseTree(tree, node, rules, pathValues):
        try:
            prevnode = tree[2].index(node)           
            leftright = '<='
            pathValues.append(values[prevnode])
        except ValueError:
            # failed, so find it as a right node - if this also causes an exception, something's really f'd up
            prevnode = tree[3].index(node)
            leftright = '>'
            pathValues.append(values[prevnode])

        # now let's get the rule that caused prevnode to -> node
        p1 = df.columns[tree[0][prevnode]]    
        p2 = tree[1][prevnode]    
        rules.append(str(p1) + ' ' + leftright + ' ' + str(p2))

        # if we've not yet reached the top, go up the tree one more step
        if prevnode != 0:
            RevTraverseTree(tree, prevnode, rules, pathValues)

    # get the nodes which are leaves
    leaves = dtc.tree_.children_left == -1
    leaves = np.arange(0,dtc.tree_.node_count)[leaves]

    # build a simpler tree as a nested list: [split feature, split threshold, left node, right node]
    thistree = [dtc.tree_.feature.tolist()]
    thistree.append(dtc.tree_.threshold.tolist())
    thistree.append(dtc.tree_.children_left.tolist())
    thistree.append(dtc.tree_.children_right.tolist())

    # get the decision rules for each leaf node & apply them
    for (ind,nod) in enumerate(leaves):

        # get the decision rules
        rules = []
        pathValues = []
        RevTraverseTree(thistree, nod, rules, pathValues)

        rules = list(reversed(rules))

        rules_list.append(rules)
        leaf_list.append(nod)
        
    df = pd.DataFrame({'leaf':leaf_list, 'rules': rules_list})
    
    pred_df = master_df
    pred_df['prediction'] = dtc.predict(X)
    pred_df['leaf'] = dtc.apply(X)
    pred_df['correct'] = (pred_df['survived'] == pred_df['prediction']).astype('int')
    pred_df['num_correct'] = pred_df.groupby('leaf').correct.transform('sum')
    pred_df['total'] = pred_df.groupby('leaf').correct.transform('count')
    pred_df['purity'] = pred_df['num_correct'] / pred_df['total']
    pred_df['majority_class'] = pred_df.groupby('leaf').survived.transform(lambda x: x.value_counts().index[0])
    pred_df = pred_df.sort_values('purity', ascending=False)
    
    pred_df = pred_df.merge(df, left_on="leaf", right_on='leaf', how='inner')
    rules_df = pred_df.groupby(['leaf', 'total', 'purity', 'majority_class']).rules.last().reset_index()
    rules_df = rules_df.sort_values(['purity', 'total'], ascending=False).reset_index(drop=True)

    return rules_df