from typing import Optional, Union, List, Any, Tuple
from omegaconf import DictConfig, ListConfig
import jax.numpy as jnp
from flax import nnx
import jax
import ast

def to_basic_types(obj):
    """
    Recursively converts JAX arrays, OmegaConf DictConfig and ListConfig to basic Python types (lists, dicts, tuples).
    This is useful for saving configurations or results in JSON format, which does not support JAX arrays or OmegaConf objects.
    """
    if isinstance(obj, jnp.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple, ListConfig)):
        return [to_basic_types(o) for o in obj]
    if isinstance(obj, DictConfig):
        return {k: to_basic_types(v) for k, v in obj.items()}
    if isinstance(obj, dict):
        return {k: to_basic_types(v) for k, v in obj.items()}
    return obj


def to_list(x: Optional[Union[Tuple[Any, ...], List[Any], Any]]) -> List[Any]:
    """
    Converts an element or a None to a list.
    If x is already a list, returns x.
    """
    if x is None:
        return []
    if isinstance(x, tuple):
        return list(x)
    return x if isinstance(x, list) else [x]

def to_tuple(x: Optional[Union[Tuple[Any, ...], List[Any], Any]]) -> Tuple[Any, ...]:
    """
    Converts an element or a None to a tuple.
    If x is already a tuple, returns x.
    """
    if x is None:
        return ()
    if isinstance(x, list):
        return tuple(x)
    return x if isinstance(x, tuple) else (x,)

def state_to_dict(state: Any) -> Any:
    """Recursively converts an nnx.State mapping into a standard Python dictionary."""
    if not isinstance(state, nnx.State):
        return state
    return {k: state_to_dict(v) for k, v in state.items()}

# Parser to convert a math string into a JAX-compatible function with maximum security, using Python's native AST to prevent any code execution.
allowed_funcs = {
    "sin": jnp.sin, "cos": jnp.cos, "tan": jnp.tan, "exp": jnp.exp, 
    "log": jnp.log, "sqrt": jnp.sqrt, "sum": jnp.sum, "mean": jnp.mean, 
    "dot": jnp.dot, "abs": jnp.abs, "sigmoid": jax.nn.sigmoid,
    "maximum": jnp.maximum, "minimum": jnp.minimum, "where": jnp.where,
}
allowed_ops = {
    ast.Add: jnp.add, ast.Sub: jnp.subtract, ast.Mult: jnp.multiply,
    ast.Div: jnp.divide, ast.Pow: jnp.power, ast.USub: jnp.negative,
    ast.BitAnd: jnp.bitwise_and, ast.BitOr: jnp.bitwise_or, ast.BitXor: jnp.bitwise_xor,
    ast.Invert: jnp.invert
} 
allowed_comps = {
    ast.Eq: jnp.equal, ast.NotEq: jnp.not_equal,
    ast.Lt: jnp.less, ast.LtE: jnp.less_equal,
    ast.Gt: jnp.greater, ast.GtE: jnp.greater_equal
}

def get_function_from_string(f_string: str) -> callable:
    """
    Parses a math string into a JAX-compatible function with maximum security.
    Uses Python's native AST to prevent any code execution.
    """
    try:
        tree = ast.parse(f_string, mode='eval').body
    except SyntaxError as e:
        raise ValueError(f"Invalid syntax in formula: '{f_string}'") from e

    return lambda theta, x: evaluate_node(tree, theta, x)

def evaluate_node(node, theta, x):
        # Handle variables (theta, x)
        if isinstance(node, ast.Name):
            if node.id == 'theta': return theta
            if node.id == 'x': return x
            if node.id == 'e': return jnp.e
            if node.id == 'pi': return jnp.pi
            if node.id in ('inf', 'infty'): return jnp.inf
            if node.id == 'nan': return jnp.nan
            raise ValueError(f"Unauthorized variable used: {node.id}")
            
        # Handle literal numbers (e.g., 1.5, 2)
        elif isinstance(node, ast.Constant):
            return node.value
            
        # Handle binary operations (e.g., a + b, a * b)
        elif isinstance(node, ast.BinOp):
            left = evaluate_node(node.left, theta, x)
            right = evaluate_node(node.right, theta, x)
            return allowed_ops[type(node.op)](left, right)
            
        # Handle unary operations (e.g., -a)
        elif isinstance(node, ast.UnaryOp):
            operand = evaluate_node(node.operand, theta, x)
            return allowed_ops[type(node.op)](operand)
            
        # Handle function calls (e.g., sin(a))
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError(f"Complex function calls not allowed")
            func_name = node.func.id
            if func_name not in allowed_funcs:
                raise ValueError(f"Unauthorized function used: {func_name}")
            args = [evaluate_node(arg, theta, x) for arg in node.args]
            return allowed_funcs[func_name](*args)
            
        # Handle array indexing (e.g., x[0])
        elif isinstance(node, ast.Subscript):
            target = evaluate_node(node.value, theta, x)
            index = evaluate_node(node.slice, theta, x)
            return target[index]
        
        # Handle comparisons (e.g., x > 0, theta == 1)
        elif isinstance(node, ast.Compare):
            if len(node.ops) > 1:
                raise ValueError("Chained comparisons (e.g. 0 < x < 1) are not supported.")
            
            left = evaluate_node(node.left, theta, x)
            right = evaluate_node(node.comparators[0], theta, x)
            return allowed_comps[type(node.ops[0])](left, right)
            
        # If the user tries to use any other Python feature (lists, dicts, imports, OS commands)
        else:
            raise TypeError(f"Unauthorized operation or syntax: {type(node).__name__}")




# Test for correctness and speed of the AST-based function against a standard lambda function using JAX for numerical operations.
# TODO: remove this from here and maybe place it in test folder.
if __name__ == "__main__":
    import os
    import timeit
    import jax
    import jax.numpy as jnp
    import ast

    if 'JAX_PLATFORMS' not in os.environ:
        try:
            from jax import devices
            if not any(d.platform == 'gpu' for d in devices()): 
                os.environ['JAX_PLATFORMS'] = 'cpu'
        except Exception:
            os.environ['JAX_PLATFORMS'] = 'cpu'

    # 1. Define the formulas
    f_string = "dot(x, theta) + sin(x[0]) + (-x[0]) - 0.5 * exp(theta[1])"
    f_ast = get_function_from_string(f_string)
    
    # Standard lambda for baseline comparison
    f_lambda = lambda theta, x: jnp.dot(x, theta) + jnp.sin(x[0]) + jnp.negative(x[0]) - 0.5 * jnp.exp(theta[1])

    # 2. Setup inputs
    theta = jnp.array([1.0, 2.0])
    x = jnp.array([0.5, 1.5])

    # 3. Verify correctness
    output_ast = f_ast(theta, x)
    output_lambda = f_lambda(theta, x)
    print(f"Output of the AST function: {output_ast}")
    print("Test passed!\n" if jnp.isclose(output_ast, output_lambda) else "Test failed!\n")

    # ==========================================
    # BENCHMARK 1: Uncompiled (Raw Python Overhead)
    # ==========================================
    def run_ast():
        f_ast(theta, x).block_until_ready()

    def run_lambda():
        f_lambda(theta, x).block_until_ready()

    n_runs = 10000
    print(f"--- Uncompiled Benchmark ({n_runs} runs) ---")
    time_ast = timeit.timeit(run_ast, number=n_runs)
    time_lambda = timeit.timeit(run_lambda, number=n_runs)

    print(f"AST Evaluator:   {time_ast:.4f} seconds")
    print(f"Standard Lambda: {time_lambda:.4f} seconds")
    print(f"Penalty:         {time_ast / time_lambda:.2f}x slower\n")

    # ==========================================
    # BENCHMARK 2: JIT Compiled (Production Speed)
    # ==========================================
    f_ast_jit = jax.jit(f_ast)
    f_lambda_jit = jax.jit(f_lambda)

    # Run once to trigger JAX compilation (warmup)
    f_ast_jit(theta, x).block_until_ready()
    f_lambda_jit(theta, x).block_until_ready()

    def run_ast_jit():
        f_ast_jit(theta, x).block_until_ready()

    def run_lambda_jit():
        f_lambda_jit(theta, x).block_until_ready()

    print(f"--- JIT Compiled Benchmark ({n_runs} runs) ---")
    time_ast_jit = timeit.timeit(run_ast_jit, number=n_runs)
    time_lambda_jit = timeit.timeit(run_lambda_jit, number=n_runs)

    print(f"JIT AST Evaluator:   {time_ast_jit:.4f} seconds")
    print(f"JIT Standard Lambda: {time_lambda_jit:.4f} seconds")
