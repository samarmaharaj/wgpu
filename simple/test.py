import cProfile
import numpy as np
from dipy.denoise.localpca import mppca

data = np.random.rand(30, 30, 30, 64).astype(np.float32)
cProfile.run("mppca(data)", sort="cumulative")

"""
         1455135 function calls (1455133 primitive calls) in 28.498 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000   28.498   28.498 {built-in method builtins.exec}
        1    0.001    0.001   28.498   28.498 <string>:1(<module>)
      2/1    0.000    0.000   28.497   28.497 decorators.py:132(wrapper)
      2/1    0.003    0.002   28.496   28.496 decorators.py:139(convert_positional_to_keyword)
        1    0.000    0.000   28.496   28.496 localpca.py:501(mppca)
        1    4.364    4.364   28.492   28.492 localpca.py:180(genpca)
    17576    0.465    0.000   18.847    0.001 _util.py:1149(wrapper)
    17576   16.516    0.001   18.051    0.001 _decomp.py:284(eigh)
    52728    2.079    0.000    2.079    0.000 {method 'dot' of 'numpy.ndarray' objects}       
    36361    0.217    0.000    2.036    0.000 fromnumeric.py:3699(mean)
    36361    0.791    0.000    1.819    0.000 _methods.py:115(_mean)
    17576    0.423    0.000    1.569    0.000 localpca.py:38(_pca_classifier)
    71515    0.866    0.000    0.866    0.000 {method 'reduce' of 'numpy.ufunc' objects}      
    17576    0.249    0.000    0.691    0.000 lapack.py:1025(_compute_lwork)
    17576    0.129    0.000    0.661    0.000 _util.py:423(_asarray_validated)
    36361    0.366    0.000    0.411    0.000 _methods.py:73(_count_reduce_items)
    17576    0.165    0.000    0.407    0.000 _function_base_impl.py:603(asarray_chkfinite)   
    17576    0.082    0.000    0.405    0.000 fromnumeric.py:2304(sum)
    52728    0.063    0.000    0.403    0.000 lapack.py:1055(<genexpr>)
    35152    0.340    0.000    0.340    0.000 lapack.py:1059(_check_work_float)
    17577    0.112    0.000    0.315    0.000 fromnumeric.py:66(_wrapreduction)
    17576    0.118    0.000    0.294    0.000 _aliases.py:70(asarray)
    17576    0.028    0.000    0.230    0.000 {method 'all' of 'numpy.ndarray' objects}       
    17576    0.024    0.000    0.202    0.000 _methods.py:64(_all)
    35152    0.200    0.000    0.200    0.000 {method 'reshape' of 'numpy.ndarray' objects}   
   124286    0.086    0.000    0.137    0.000 {built-in method builtins.isinstance}
    17576    0.079    0.000    0.120    0.000 {built-in method numpy.array}
    17576    0.069    0.000    0.100    0.000 blas.py:395(getter)
    17576    0.016    0.000    0.100    0.000 _sparse.py:10(issparse)
    17576    0.033    0.000    0.098    0.000 fromnumeric.py:604(transpose)
    17576    0.035    0.000    0.065    0.000 fromnumeric.py:48(_wrapfunc)
    17576    0.046    0.000    0.056    0.000 _helpers.py:683(_check_device)
    17576    0.016    0.000    0.051    0.000 <frozen abc>:117(__instancecheck__)
    52738    0.048    0.000    0.048    0.000 {built-in method builtins.getattr}
    17577    0.042    0.000    0.042    0.000 fromnumeric.py:67(<dictcomp>)
    35191    0.041    0.000    0.041    0.000 {method 'get' of 'dict' objects}
    17576    0.031    0.000    0.041    0.000 enum.py:193(__get__)
    17576    0.034    0.000    0.040    0.000 _type_check_impl.py:270(iscomplexobj)
    17576    0.035    0.000    0.035    0.000 {built-in method _abc._abc_instancecheck}       
    90298    0.034    0.000    0.034    0.000 {built-in method builtins.issubclass}
    36361    0.030    0.000    0.030    0.000 {built-in method numpy.lib.array_utils.normalize_axis_index}
    17576    0.019    0.000    0.026    0.000 core.py:6669(isMaskedArray)
    36361    0.021    0.000    0.021    0.000 {built-in method numpy.asanyarray}
    36361    0.019    0.000    0.019    0.000 fromnumeric.py:3694(_mean_dispatcher)
    17576    0.017    0.000    0.017    0.000 {method 'update' of 'dict' objects}
    17576    0.017    0.000    0.017    0.000 {method 'transpose' of 'numpy.ndarray' objects} 
    35172    0.015    0.000    0.015    0.000 {method 'append' of 'list' objects}
    18787    0.015    0.000    0.015    0.000 {built-in method builtins.hasattr}
    17577    0.012    0.000    0.012    0.000 {built-in method numpy.asarray}
    17576    0.012    0.000    0.012    0.000 _array_api_override.py:69(array_namespace)      
    35156    0.011    0.000    0.011    0.000 {built-in method builtins.len}
    17576    0.011    0.000    0.011    0.000 {built-in method builtins.any}
    17576    0.010    0.000    0.010    0.000 enum.py:1257(value)
    17578    0.008    0.000    0.008    0.000 {method 'items' of 'dict' objects}
    17576    0.007    0.000    0.007    0.000 fromnumeric.py:600(_transpose_dispatcher)       
    17576    0.007    0.000    0.007    0.000 _misc.py:181(_datacopied)
    17576    0.007    0.000    0.007    0.000 fromnumeric.py:2299(_sum_dispatcher)
    17576    0.006    0.000    0.006    0.000 _type_check_impl.py:171(_is_type_dispatcher)    
        1    0.000    0.000    0.004    0.004 {method 'clip' of 'numpy.ndarray' objects}      
        1    0.004    0.004    0.004    0.004 _methods.py:96(_clip)
        2    0.002    0.001    0.002    0.001 {method 'astype' of 'numpy.ndarray' objects}    
        1    0.001    0.001    0.002    0.002 localpca.py:86(create_patch_radius_arr)
        1    0.001    0.001    0.001    0.001 numeric.py:246(ones_like)
        8    0.000    0.000    0.001    0.000 version.py:47(parse)
        2    0.000    0.000    0.001    0.000 inspect.py:3261(signature)
        8    0.000    0.000    0.001    0.000 version.py:188(__init__)
        2    0.000    0.000    0.001    0.000 inspect.py:3007(from_callable)
        2    0.000    0.000    0.001    0.000 inspect.py:2435(_signature_from_callable)       
        2    0.000    0.000    0.001    0.000 inspect.py:2331(_signature_from_function)       
        1    0.000    0.000    0.000    0.000 fromnumeric.py:2436(any)
        1    0.000    0.000    0.000    0.000 fromnumeric.py:86(_wrapreduction_any_all)       
        1    0.000    0.000    0.000    0.000 <frozen abc>:121(__subclasscheck__)
        1    0.000    0.000    0.000    0.000 {built-in method _abc._abc_subclasscheck}       
       16    0.000    0.000    0.000    0.000 inspect.py:2669(__init__)
       80    0.000    0.000    0.000    0.000 {method 'group' of 're.Match' objects}
        8    0.000    0.000    0.000    0.000 version.py:523(_cmpkey)
        1    0.000    0.000    0.000    0.000 lapack.py:927(get_lapack_funcs)
        8    0.000    0.000    0.000    0.000 {method 'search' of 're.Pattern' objects}       
        2    0.000    0.000    0.000    0.000 inspect.py:2955(__init__)
        3    0.000    0.000    0.000    0.000 numeric.py:170(ones)
        1    0.000    0.000    0.000    0.000 blas.py:337(_get_funcs)
        1    0.000    0.000    0.000    0.000 localpca.py:116(compute_patch_size)
        2    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
       16    0.000    0.000    0.000    0.000 enum.py:688(__call__)
        8    0.000    0.000    0.000    0.000 version.py:511(_parse_local_version)
        2    0.000    0.000    0.000    0.000 version.py:358(base_version)
        1    0.000    0.000    0.000    0.000 localpca.py:135(compute_num_samples)
       32    0.000    0.000    0.000    0.000 version.py:207(<genexpr>)
        1    0.000    0.000    0.000    0.000 fromnumeric.py:3287(prod)
        2    0.000    0.000    0.000    0.000 {built-in method builtins.sum}
       24    0.000    0.000    0.000    0.000 version.py:471(_parse_letter_version)
        8    0.000    0.000    0.000    0.000 <string>:1(<lambda>)
       18    0.000    0.000    0.000    0.000 inspect.py:3002(<genexpr>)
        2    0.000    0.000    0.000    0.000 inspect.py:735(unwrap)
        4    0.000    0.000    0.000    0.000 decorators.py:159(<genexpr>)
        4    0.000    0.000    0.000    0.000 {method 'join' of 'str' objects}
        1    0.000    0.000    0.000    0.000 blas.py:270(find_best_blas_type)
        2    0.000    0.000    0.000    0.000 inspect.py:167(get_annotations)
        6    0.000    0.000    0.000    0.000 version.py:516(<genexpr>)
        6    0.000    0.000    0.000    0.000 version.py:578(<genexpr>)
       16    0.000    0.000    0.000    0.000 enum.py:1095(__new__)
        8    0.000    0.000    0.000    0.000 version.py:379(<genexpr>)
        3    0.000    0.000    0.000    0.000 {built-in method numpy.empty}
        4    0.000    0.000    0.000    0.000 version.py:84(__le__)
        2    0.000    0.000    0.000    0.000 {method 'split' of 're.Pattern' objects}        
        1    0.000    0.000    0.000    0.000 copy.py:128(deepcopy)
        8    0.000    0.000    0.000    0.000 {method 'split' of 'str' objects}
        8    0.000    0.000    0.000    0.000 {built-in method __new__ of type object at 0x00007FFC5CA0EBC0}
       18    0.000    0.000    0.000    0.000 version.py:537(<lambda>)
       16    0.000    0.000    0.000    0.000 {method 'isidentifier' of 'str' objects}        
       16    0.000    0.000    0.000    0.000 {method '__contains__' of 'frozenset' objects}  
        8    0.000    0.000    0.000    0.000 {method 'lower' of 'str' objects}
       16    0.000    0.000    0.000    0.000 inspect.py:2722(name)
        4    0.000    0.000    0.000    0.000 inspect.py:378(isfunction)
        1    0.000    0.000    0.000    0.000 fromnumeric.py:88(<dictcomp>)
        4    0.000    0.000    0.000    0.000 multiarray.py:1085(copyto)
       16    0.000    0.000    0.000    0.000 inspect.py:2734(kind)
        2    0.000    0.000    0.000    0.000 {method 'values' of 'mappingproxy' objects}     
        3    0.000    0.000    0.000    0.000 {built-in method builtins.id}
        4    0.000    0.000    0.000    0.000 {method 'isdigit' of 'str' objects}
        4    0.000    0.000    0.000    0.000 {built-in method builtins.callable}
        2    0.000    0.000    0.000    0.000 version.py:267(epoch)
        1    0.000    0.000    0.000    0.000 fromnumeric.py:2431(_any_dispatcher)
        2    0.000    0.000    0.000    0.000 inspect.py:3015(parameters)
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
        2    0.000    0.000    0.000    0.000 {built-in method sys.getrecursionlimit}
        2    0.000    0.000    0.000    0.000 version.py:278(release)
        1    0.000    0.000    0.000    0.000 fromnumeric.py:3282(_prod_dispatcher)
        1    0.000    0.000    0.000    0.000 numeric.py:240(_ones_like_dispatcher)
        1    0.000    0.000    0.000    0.000 multiarray.py:115(empty_like)
        1    0.000    0.000    0.000    0.000 copy.py:182(_deepcopy_atomic)

"""