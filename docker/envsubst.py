#!/usr/bin/python

import os, re, sys

def envsubst(input_str: str) -> str:
    # Only substitute variables of the form $VAR or ${VAR}
    # Do not replace $$VAR (escaped $)
    pattern = re.compile(r'(\$+)(\w+|\{\w+\})')


    def replacer(match):
        # Count the number of $ signs
        dollar_signs = match.group(1)
        if len(dollar_signs) % 2 == 0:
            # Even number of $ means escaped $, return half the $ signs and the variable name unchanged
            return dollar_signs[:len(dollar_signs)//2] + match.group(2)
        else:
            # Odd number of $ means one $ for substitution, return half the $ signs and the variable value
            var_name = match.group(2)
            if var_name.startswith('{') and var_name.endswith('}'):
                var_name = var_name[1:-1]
            
            # If environment variable is not found, print error to stderr and return empty string
            value = os.getenv(var_name)
            if value is None:
                print(f"Warning: Environment variable '{var_name}' not found.", file=sys.stderr)
                sys.exit(1) 
            return dollar_signs[:len(dollar_signs)//2] + value


    result = pattern.sub(replacer, input_str)
    return result

if __name__ == "__main__":
    input_data = sys.stdin.read()
    output_data = envsubst(input_data)
    sys.stdout.write(output_data)
