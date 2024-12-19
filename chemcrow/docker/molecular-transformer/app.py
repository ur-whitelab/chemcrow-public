import re
import subprocess
from flask import Flask, request, jsonify
from rdkit import Chem

app = Flask(__name__)


SMI_REGEX_PATTERN =  r"(\%\([0-9]{3}\)|\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\||\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"

def canonicalize_smiles(smiles, verbose=False): # will raise an Exception if invalid SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol)
    else:
        if verbose:
            print(f'{smiles} is invalid.')
        return ''

def smiles_tokenizer(smiles):
    """Canonicalize and tokenize input smiles"""
    
    smiles = canonicalize_smiles(smiles)
    smiles_regex = re.compile(SMI_REGEX_PATTERN)
    tokens = [token for token in smiles_regex.findall(smiles)]
    return ' '.join(tokens)


@app.route('/api/v1/run', methods=['POST'])
def f():
    request_data = request.get_json()
    input = request_data['smiles']

    # Write the input to 'inp.txt'
    with open('input.txt', 'w') as f:
        # Tokenize smiles
        smi = smiles_tokenizer(input)
        f.write(smi)

    model_path = 'models/USPTO480k_model_step_400000.pt'

    src_path = 'input.txt'
    output_path = 'output.txt'
    n_best = 5
    beam_size = 10
    max_length = 300
    batch_size = 1

    try:
        # Construct the command to execute
        cmd = f"onmt_translate -model {model_path} " \
              f"--src {src_path} " \
              f"--output {output_path} --n_best {n_best} " \
              f"--beam_size {beam_size} --max_length {max_length} " \
              f"--batch_size {batch_size}"

        # Execute the command using subprocess.check_call()
        subprocess.check_call(cmd, shell=True)

        # Read produced output
        with open('output.txt', 'r') as f:
            prods = f.read()
            prods = re.sub(' ', '', prods).split('\n')

        
        # Return a success message
        return jsonify({'status': 'SUCCESS', 'product': prods})

    except:
        return jsonify({'status': 'ERROR', 'product': None})

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0')

