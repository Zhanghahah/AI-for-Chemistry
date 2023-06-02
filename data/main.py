import click
from datasets import load_dataset
import os
from functools import partial
from rdkit import Chem
from rdkit.rdBase import BlockLogs
from rdkit import DataStructs
from rdkit.Chem import rdMolDescriptors

from tqdm import tqdm
import json

block = BlockLogs()

SCRIPT_ROOT = os.path.dirname(os.path.realpath(__file__))


def product_eval(yhat, y):
    m1 = Chem.MolFromSmiles(yhat)
    m2 = Chem.MolFromSmiles(y)
    # return tanimoto similarity
    return DataStructs.TanimotoSimilarity(
        rdMolDescriptors.GetMorganFingerprintAsBitVect(m1, 2),
        rdMolDescriptors.GetMorganFingerprintAsBitVect(m2, 2),
    )


def valid_mol_eval(yhat, prompt):
    try:
        m1 = Chem.MolFromSmiles(prompt + yhat)
    except:
        return 0
    if m1 is None:
        return 0
    return 1


def product_task(full=False):
    """
    Generator that produces a tuple of prompt and evaluation function. The evaluation function takes a string and returns a similarity score (higher is better)
    """
    data = load_dataset(
        f"{SCRIPT_ROOT}/ord.py", "full" if full else "small", split="train",
        # streaming=True
    )

    json_file = open(
        os.path.join(f"{SCRIPT_ROOT}/", 'train_v1.json'),
                    'w',
                    encoding='utf-8')

    llama_file = open(
        os.path.join(f"{SCRIPT_ROOT}/", 'llama_chem_v1.json'),
                    'w',
                    encoding='utf-8')

    for i, example in enumerate(data):
        output = dict()
        llama_output = dict()
        try:
            reactants, reagents, \
            solvents, catalysts, products = example["text"].split("||")
        except:
            print(example["text"])
            pass

        rxn_name = example["rxn_name"]
        reaction_id = example["reaction_id"]
        conditions = example["notes"]
        rxn_yields = example["yield"]
        reference = example["rxn_reference"]

        # REAGENTS = ""
        if reagents == "":
            REAGENTS = "this reaction does not need reagents,"
        else:
            REAGENTS = f"Reagents are {reagents},"

        # SOLVENTS = ""
        if solvents == "":
            SOLVENTS = "this reaction does not need solvents,"
        else:
            SOLVENTS = f"Solvents are {solvents},"

        # CATALYSTS = ""
        if catalysts == "":
            CATALYSTS = "this reaction does not need catalysts,"
        else:
            CATALYSTS = f"Catalysts are {catalysts}."


        llama_output["INSTRUCTION"] = f"Here is a chemical reaction formula: " + \
                                        REAGENTS + \
                                        SOLVENTS + \
                                        CATALYSTS + \
                                      f"Products are {products}, " \
                                      f"please give me the reaction condition of this chemical formula."

        llama_output["RESPONSE"] = f"The condition of this chemical reaction is: {conditions}"

        output["reactants"] = reactants
        output["solvents"] = solvents
        output["catalysts"] = catalysts
        output["reagents"] = reagents
        output["products"] = products
        output["reaction_name"] = rxn_name
        output["reaction_id"] = reaction_id
        output["conditions"] = conditions
        output["yield"] = rxn_yields
        output["reference"] = reference

        products_list = products.split(";")
        yields_list = rxn_yields.split(";")

        try:
            assert len(products_list) == len(yields_list)
        except:
            print(products_list)
            pass
        try:
            updated_yields = ["0" if yield_value.split(":")[-1] == "None" else yield_value.split(":")[-1]
                              for idx, yield_value in enumerate(yields_list)]
            products_yields = sorted(list(zip(products_list, updated_yields)),
                                                        key=lambda x: -float(x[1]))
        except:
            print(yields_list)
        main_product = products_yields[0][0]
        main_product_yield = products_yields[0][1]
        output["main_product"] = main_product.split(":")[-1]
        output["main_product_id"] = main_product.split(":")[0]
        output["main_product_yield"] = main_product_yield

        json_str = json.dumps(output, ensure_ascii=False) + "\n"
        json_file.write(json_str)

        llama_json_str = json.dumps(llama_output, ensure_ascii=False) + "\n"
        llama_file.write(llama_json_str)



def valid_mol_task(full=False):
    """
    Generator that produces a tuple of prompt and evaluation function. The evaluation function takes a string and returns a success score (1 or 0)
    """
    data = load_dataset(
        f"{SCRIPT_ROOT}/coconut.py", "full" if full else "small", split="train"
    )
    output = []
    for i, example in enumerate(data):
        mol = example["text"]
        # split in half
        s = mol[: len(mol) // 2]
        yield s, partial(valid_mol_eval, prompt=s)


if __name__ == "__main__":

    product_task(full=True)
        # break

    # for prompt, eval in valid_mol_task():
    #     print(prompt)
    #     print(eval("Pi"))
    #     break
