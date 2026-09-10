import picwavelib as picw
import sys

def test_syntax(keyword):
    print(f"Testing keyword: {keyword}")
    # Read the existing template file
    with open("custom_refbase_template.txt", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Replace transition 0 4
    target = 'BEGIN_TRANSITION 0 4    // DS â†” WL                         INDEX 4\n   NRADRATE_U tau_r_W 0 0\nEND_TRANSITION'
    replacement = f'BEGIN_TRANSITION 0 4    // DS \u2194 WL                         INDEX 4\n   {keyword} "if(N<1.602637e+18, poly1d((N-1.057953e+17)/1.496842e+18, -2.648745e+24, 5.114014e+25, -1.322730e+26, 2.784289e+26), 1.946473e+26+1.181767e+09*max(0,N-1.602637e+18)**0.9763)*20"\nEND_TRANSITION'
    
    # Replace also general baseline values
    # GS/ES1/ES2 template parameters to some numbers
    import numpy as np
    from LI_optimiser import BASELINE, get_full_params
    log2_zeros = np.zeros(18)
    params, spon = get_full_params(log2_zeros)
    
    # Re-write content with replaced template parameters
    text = content.replace(target, replacement)
    for name in sorted(params.keys(), key=len, reverse=True):
        if isinstance(params[name], (int, float)):
            text = text.replace(name, f"{params[name]:.4e}")
            
    with open("test_syntax.mat", "w", encoding="utf-8") as f:
        f.write(text)
        
    # Connect and load
    try:
        app = picw.connect_to_picwave()
        circuit = app.getsubnode("subnodes[1].subnodes[2]")
        circuit.setmaterbase("[PrjDir]\\test_syntax.mat")
        # Run a short step
        circuit.tdcalculator.run()
        print("Success! Simulation ran fine.")
        picw.disconnect_picwave()
        return True
    except Exception as e:
        print(f"Failed with error: {e}")
        try:
            picw.disconnect_picwave()
        except:
            pass
        return False

# Test NRADRECOMB_EXPRESSION first
success = test_syntax("NRADRECOMB_EXPRESSION")
if not success:
    print("Trying NRADRATE_U_EXPRESSION...")
    test_syntax("NRADRATE_U_EXPRESSION")
