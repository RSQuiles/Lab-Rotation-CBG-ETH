import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import os
import pickle 
import ast
import requests
from Bio import Entrez
import sys
from tqdm import tqdm

def open_xml(datapath, pathway):
    xml_file = datapath + pathway + '.xml'
    tree = ET.parse(xml_file)
    root = tree.getroot()
    return tree, root

def parse_xml(tree, root):
    genes = {} # Dictionary of id:name pairs for genes
    compounds = {} # Dictionary of id:name pairs for compounds
    groups = {} # Dictionary of id:component(list) pairs for groups
    relations = pd.DataFrame(columns=['source', 'target', 'type', 'subtype']) # DataFrame for relations
    #node_file = pathway + '_nodes.att'
    genes_file = datapath + pathway + '_genes.pkl'
    genes_saved = False
    if os.path.exists(genes_file):
        genes_saved = True
        print("Loading genes from file.")
        with open(genes_file, 'rb') as f:
            genes = pickle.load(f)
    for entry in tqdm(root.findall('.//entry'), desc='Parsing XML'):
        # Extract genes
        if entry.get('type') == 'gene':
            if genes_saved:
                continue
            kegg_id = entry.get('id')
            entrez_id = entry.get('name')
            genes[kegg_id] = entrez_id
        # Extract compounds
        elif entry.get('type') == 'compound':
            id = entry.get('id')   
            name = entry.get('name')
            compounds[id] = name
        # Extract groups (necessary?)
        elif entry.get('type') == 'group':
            id = entry.get('id')
            components = []
            for component in entry.findall('./component'):
                comp_id = component.get('id')
                components.append(comp_id)
            groups[id] = components
    # Extract relations
    for relation in root.findall('.//relation'):
        source = relation.get('entry1')
        target = relation.get('entry2')
        type = relation.get('type')
        if relation.find('./subtype') is None:
            subtype = 'NA'
        else:
            subtype = relation.find('./subtype').get('name')
        relations = pd.concat([relations, pd.DataFrame([[source, target, type, subtype]], columns=['source', 'target', 'type', 'subtype'])], ignore_index=True)
    
    return genes, compounds, groups, relations

def retrieve_annotations(genes_list):
    request = Entrez.epost("gene", id=",".join(genes_list))
    try:
        result = Entrez.read(request)
    except RuntimeError as e:
        # FIXME: How generate NAs instead of causing an error with invalid IDs?
        print(f"The following error occurred while retrieving the annotations: {e}")
        sys.exit(-1)

    webEnv = result["WebEnv"]
    queryKey = result["QueryKey"]
    data = Entrez.esummary(db="gene", webenv=webEnv, query_key=queryKey)
    try: 
        annotations = Entrez.read(data)
    except RuntimeError as e:
        print(f"Could not read data: {data.read()}")
        print(f"The following error occurred while reading the annotations: {e}")
        sys.exit(-1)

    print(annotations)

    #print(f"Retrieved {len(annotations)} annotations for {len(genes_list)} genes")

    return annotations

def retrieve_gene_symbols(genes_list):
    annotations = retrieve_annotations(genes_list)
    # Extract gene symbols from the annotations
    names = []
    try:
    # Navigate to the 'Name' field in the dictionary
        names = [doc['Name'] for doc in annotations['DocumentSummarySet']['DocumentSummary']]
    except (KeyError, IndexError, TypeError) as e:
        print(f"Error extracting 'Name': {e}")
        return None
    #Turn list into single string
    return names

def save_nodes(genes, compounds, groups, pathway, datapath):
    node_file = datapath + pathway + '_nodes.att'
    
    if os.path.exists(node_file):
        with open(node_file) as f:
            nodes = pd.read_csv(f, sep='\t', names=['kegg_id', 'entrez_id', 'gene_symbols', 'node_type', 'pathway'])
        nodes['entrez_id'] = nodes['entrez_id'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else [] if pd.isna(x) else x
        )
        nodes['gene_symbols'] = nodes['gene_symbols'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else [] if pd.isna(x) else x
        )
        #print(nodes)
    
    else:
        nodes = pd.DataFrame(columns=['kegg_id', 'entrez_id', 'gene_symbols', 'node_type', 'pathway'])
        for group in tqdm(groups.items(), desc='Processing groups'):
            kegg_id_group = group[0]
            #print(f"Processing group {group[0]}...")
            entrez_ids = []
            compound_ids = []
            for kegg_id_component in group[1]:
                if kegg_id_component in genes.keys():
                    entrez_ids.extend(genes[kegg_id_component].split(' '))
                else:
                    compound_ids.append(compounds[kegg_id_component])
            entrez_ids = [id[4:] for id in entrez_ids if id.startswith('hsa:')]
            compound_ids = [id[4:] for id in compound_ids if id.startswith('cpd:')]
            #print(f"Entrez IDs: {entrez_ids}")
            gene_symbols = retrieve_gene_symbols(entrez_ids)
            if gene_symbols is None:
                gene_symbols = []
                print(f"Could not retrieve gene symbols for {entrez_ids}")
            #Add whole group
            #Obs: entrez_id column contains the kegg_ids of the components to make later processing easier
            nodes = pd.concat([nodes, pd.DataFrame([[kegg_id_group, group[1], gene_symbols, 'complex', pathway]], columns=['kegg_id', 'entrez_id', 'gene_symbols', 'node_type', 'pathway'])], ignore_index=True)
            
        for kegg_id, entrez_id in tqdm(genes.items(), desc='Processing genes'):
            #print(f"Processing gene {kegg_id}...")
            entrez_ids = entrez_id.split(' ')
            entrez_ids = [id[4:] for id in entrez_ids]
            #print(f"Entrez IDs: {entrez_ids}")
            if len(entrez_ids) > 1:
                node_type = 'protein family'
            else: 
                node_type = 'simple gene'
            gene_symbols = retrieve_gene_symbols(entrez_ids)
            if len(entrez_ids) != len(gene_symbols):
                print(f'{len(gene_symbols)} gene symbols for {node_type} with {len(entrez_ids)} entrez ids')
            nodes = pd.concat([nodes, pd.DataFrame([[kegg_id, entrez_ids, gene_symbols, node_type, pathway]], columns=['kegg_id', 'entrez_id', 'gene_symbols', 'node_type', 'pathway'])], ignore_index=True)

        for compound in tqdm(compounds.items(), desc='Processing compounds'):
            cpd_ids = compound[1].split(' ')
            cpd_ids = [id[4:] for id in cpd_ids]
            nodes = pd.concat([nodes, pd.DataFrame([[compound[0], cpd_ids, 'compound', 'compound', pathway]], columns=['kegg_id', 'entrez_id', 'gene_symbols', 'node_type', 'pathway'])], ignore_index=True)

        nodes = uniquefy_df(nodes)

        with open(node_file, 'w') as f:
            for index, row in nodes.iterrows():
                f.write(f"{row['kegg_id']}\t{row['entrez_id']}\t{row['gene_symbols']}\t{row['node_type']}\t{row['pathway']}\n")

    return nodes

def get_colnames(nodes):
    column_list = []
    for index, row in nodes.iterrows():
        gene_symbol_list = row['gene_symbols']
        #Handle nans
        if isinstance(gene_symbol_list, list):
            #gene_symbol_str = "-".join(gene_symbol_list)
            for symbol in gene_symbol_list:
                column_list.append(symbol)
        elif pd.isna(gene_symbol_list):
            print(f"NaN value found in gene symbols for {row['kegg_id']}.")
        elif gene_symbol_list == 'compound':
            continue
        else:
            print(f"Unknown formatting for gene symbols: {gene_symbol_list}.")
    # Remove duplicates
    column_list = list(set(column_list))
    
    return column_list
       

def save_edges(relations, pathway, datapath):
    edge_file = datapath + pathway + '_edges.sif'
    
    if os.path.exists(edge_file):
        with open(edge_file) as f:
            edges = pd.read_csv(f, sep='\t', names=['kegg_id1', 'interaction', 'kegg_id2', 'pathway'])
    
    else:
        edges = pd.DataFrame(columns=['kegg_id1', 'interaction', 'kegg_id2', 'pathway'])

        edges = edges.assign(
            kegg_id1=relations['source'],
            interaction=relations['subtype'],
            kegg_id2=relations['target'],
            pathway=pathway
        )

        with open(edge_file, 'w') as f:
            for index, row in edges.iterrows():
                f.write(f"{row['kegg_id1']}\t{row['interaction']}\t{row['kegg_id2']}\t{row['pathway']}\n")

    return edges

def find_endnodes_startnodes(nodes, edges):
    # Find endnodes and startnodes
    endnode_ids = set(edges['kegg_id2']) - set(edges['kegg_id1'])
    endnodes = nodes[nodes['kegg_id'].isin(endnode_ids)]
    startnode_ids = set(edges['kegg_id1']) - set(edges['kegg_id2'])
    startnodes = nodes[nodes['kegg_id'].isin(startnode_ids)]
    return endnodes, startnodes

def get_circuit(nodes, edges, node_circuit, startnode, visited=None, direction='backward'):
    #print(f"Getting circuit for {startnode['kegg_id']} ({startnode['node_type']})")
    downstream_node_circuit = pd.DataFrame()

    if visited is None:
        visited = set()
    if startnode['kegg_id'] in visited:
        return node_circuit
    visited.add(startnode['kegg_id'])

    # Add current node
    if startnode['node_type'] == 'simple gene':
        downstream_node_circuit = pd.concat([node_circuit, pd.DataFrame([startnode])], ignore_index=True)
    if startnode['node_type'] == 'protein family':
        downstream_node_circuit = pd.concat([node_circuit, pd.DataFrame([startnode])], ignore_index=True)

    # Recurse for complexes
    elif startnode['node_type'] == 'complex':
        # Get all components of the complex
        for kegg_id_component in startnode['entrez_id']:
            matching_nodes = pd.DataFrame(columns=nodes.columns)
            for index, node in nodes.iterrows():
                #print(f"Checking {node['kegg_id']} (type {type(node['kegg_id'])}) against {kegg_id_component} (type {type(kegg_id_component)}) in groups")
                if str(node['kegg_id']).strip() == kegg_id_component:
                    matching_nodes = pd.concat([matching_nodes, pd.DataFrame([node])], ignore_index=True)
            if matching_nodes.shape[0] != 1:
                print(f"{matching_nodes.shape[0]} matching nodes for KEGG ID {kegg_id_component}") 
            for index, matching_node in matching_nodes.iterrows():
                component_node = get_circuit(nodes, edges, node_circuit, matching_node, visited=visited, direction=direction)
                #if component_node.shape[0] == 1:
                    #print(f"No circuit upstream/downstream for node {matching_node['kegg_id']}")
                #Concatenate
                pre = downstream_node_circuit.shape[0]
                #print(f'Groups: Concatenating {component_node.shape} to {downstream_node_circuit.shape}')
                if downstream_node_circuit.shape[0] == 0:
                    downstream_node_circuit = component_node
                else:
                    downstream_node_circuit = pd.concat([downstream_node_circuit, component_node], ignore_index=True)
                if pre + component_node.shape[0] != downstream_node_circuit.shape[0]:
                    print(f"Error in concatenation of groups while getting circuit for {startnode['kegg_id']} ({startnode['node_type']})")

    # Get previous or next nodes
    if direction == 'backward':
        edge_list = list(edges.loc[edges['kegg_id2'] == startnode['kegg_id'], 'kegg_id1'])
    elif direction == 'forward':
        edge_list = list(edges.loc[edges['kegg_id1'] == startnode['kegg_id'], 'kegg_id2'])

    # Recurse for previous/next nodes (even if complex)
    for kegg_id in edge_list:
        #print(f"Checking {kegg_id} (type {type(kegg_id)} against {startnode['kegg_id']} (type {type(startnode['kegg_id'])})")
        node = nodes.loc[nodes['kegg_id'] == kegg_id].iloc[0]
        assert pd.DataFrame([node]).shape[0] == 1, f"Node {node} not found in nodes DataFrame"
        branch = get_circuit(nodes, edges, node_circuit, node, visited=visited, direction=direction)
        #if branch.shape[0] == 1:
            #print(f"No circuit upstream/downstream for node {node['kegg_id']}")
        #Concatenate
        pre = downstream_node_circuit.shape[0]
        #print(f'Branches: Concatenating {branch.shape} to {downstream_node_circuit.shape}')
        if downstream_node_circuit.shape[0] == 0:
            downstream_node_circuit = branch
        else:
            downstream_node_circuit = pd.concat([downstream_node_circuit, branch], ignore_index=True)
        if pre + branch.shape[0] != downstream_node_circuit.shape[0]:
            print(f"Error in concatenation of branches while getting circuit for {startnode['kegg_id']} ({startnode['node_type']})")

    return downstream_node_circuit

def uniquefy_df(df):
    df_processed = df.map(lambda x: tuple(x) if isinstance(x, list) else x)
    df_processed = df_processed.drop_duplicates()
    df_processed = df_processed.map(lambda x: list(x) if isinstance(x, tuple) else x)
    return df_processed

def build_matrixes(nodes, edges, all_endnodes, all_startnodes, column_list, datapath, pathway):
    # Create a DataFrames for the circuit matrixes with all zeros
    genes_per_circuit_backward = pd.DataFrame(columns=column_list)
    genes_per_circuit_forward = pd.DataFrame(columns=column_list)
    # Create a DataFrame for the genes per pathways matrix with all ones
    genes_per_pathway = pd.DataFrame(np.ones((1, len(column_list))), columns=column_list, index=[pathway])
    
    for index, endnode in tqdm(all_endnodes.iterrows(), desc='Building genes per circuits backwards matrix'):
        # Get circuits from endnodes to startnodes
        nodes_in_circuit_backward = pd.DataFrame()
        #print(f'Endnode: {endnode["kegg_id"]} ({endnode["node_type"]})')
        nodes_in_circuit_backward = get_circuit(nodes, edges, nodes_in_circuit_backward, endnode, direction='backward')
        #print(nodes_in_circuit_backward)
        nodes_in_circuit_backward = uniquefy_df(nodes_in_circuit_backward)

        # Fill out genes per circuits backwards matrix
        circuit_name = pathway + '_' + "-".join(endnode['gene_symbols']) + '_backward'
        #pathway_annotation = input(f"Pathway annotation for {circuit_name}:")
        #circuit_name = circuit_name + '_' + pathway_annotation
        genes_per_circuit_backward_row = pd.DataFrame(np.zeros((1, len(column_list))), columns=column_list, index=[circuit_name])
        for index, node in nodes_in_circuit_backward.iterrows():
            for symbol in node['gene_symbols']:
                if symbol in genes_per_circuit_backward_row.columns:
                    genes_per_circuit_backward_row[symbol] = 1
                else:
                    print(f"Gene symbol {symbol} not found in column list.")
        genes_per_circuit_backward_row.index = [circuit_name]
        if genes_per_circuit_backward.shape[0] == 0:
            genes_per_circuit_backward = genes_per_circuit_backward_row
        else:
            genes_per_circuit_backward = pd.concat([genes_per_circuit_backward, genes_per_circuit_backward_row], ignore_index=False)

    for index, startnode in tqdm(all_startnodes.iterrows(), desc='Building genes per circuits forwards matrix'):
        # Get circuits from startnodes to endnodes
        nodes_in_circuit_forward = pd.DataFrame()
        nodes_in_circuit_forward = get_circuit(nodes, edges, nodes_in_circuit_forward, startnode, direction='forward')
        nodes_in_circuit_forward = uniquefy_df(nodes_in_circuit_forward)

        # Fill out genes per circuits forwards matrix
        circuit_name = pathway + '_' + '-'.join(startnode['gene_symbols']) + '_forward'
        genes_per_circuit_forward_row = pd.DataFrame(np.zeros((1, len(column_list))), columns=column_list, index=[circuit_name])
        for index, node in nodes_in_circuit_forward.iterrows():
            for symbol in node['gene_symbols']:
                if symbol in genes_per_circuit_forward_row.columns:
                    genes_per_circuit_forward_row[symbol] = 1
                else:
                    print(f"Gene symbol {symbol} not found in column list.")
        genes_per_circuit_forward_row.index = [circuit_name]
        if genes_per_circuit_forward.shape[0] == 0:
            genes_per_circuit_forward = genes_per_circuit_forward_row
        else:
            genes_per_circuit_forward = pd.concat([genes_per_circuit_forward, genes_per_circuit_forward_row], ignore_index=False)
    
    return genes_per_pathway, genes_per_circuit_backward, genes_per_circuit_forward

if __name__ == "__main__":
    # Definitions
    Entrez.email = 'mazevedo@student.ethz.ch'
    datapath = './data/KEGG/'
    pathways_file = datapath + 'pathways.txt'
    circuit_backwards_file = datapath + 'genes_per_circuit_backward.csv'
    circuit_forwards_file = datapath + 'genes_per_circuit_forward.csv'
    pathway_matrix_file = datapath + 'genes_per_pathway.csv'
    # get_subpathways = False

    # Load files
    with open(pathways_file, 'rb') as f:
        pathways_list = pickle.load(f)
    # genes_per_pathways_total = pd.read_csv(pathway_matrix_file, sep=',')

    genes_per_pathways_total = pd.DataFrame()
    genes_per_circuit_backward_total = pd.DataFrame()
    genes_per_circuit_forward_total = pd.DataFrame()

    for pathway in pathways_list:
        print(f"Processing pathway {pathway}...")
        tree, root = open_xml(datapath, pathway)
        genes, compounds, groups, relations = parse_xml(tree, root)
        nodes = save_nodes(genes, compounds, groups, pathway, datapath)
        edges = save_edges(relations, pathway, datapath)
        print(nodes)
        column_list = get_colnames(nodes)
        endnodes, startnodes = find_endnodes_startnodes(nodes, edges)
        genes_per_pathways, genes_per_circuit_backward, genes_per_circuit_forward = build_matrixes(nodes, edges, 
                                                                                                   endnodes, startnodes, 
                                                                                                   column_list,
                                                                                                   datapath, pathway)
        genes_per_pathways_total = pd.concat([genes_per_pathways_total, genes_per_pathways], ignore_index=False).fillna(0)
        genes_per_circuit_backward_total = pd.concat([genes_per_circuit_backward_total, genes_per_circuit_backward], ignore_index=False).fillna(0)
        genes_per_circuit_forward_total = pd.concat([genes_per_circuit_forward_total, genes_per_circuit_forward], ignore_index=False).fillna(0)

    # Save matrixes to csv
    genes_per_circuit_backward_total.to_csv(circuit_backwards_file, sep=',', index=True)
    genes_per_circuit_forward_total.to_csv(circuit_forwards_file, sep=',', index=True)
    genes_per_pathways_total.to_csv(pathway_matrix_file, sep=',', index=True)