function flag = allclose(a, b)
% https://numpy.org/doc/stable/reference/generated/numpy.allclose.html
% numpy.allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)
% https://stackoverflow.com/questions/28975822/matlab-equivalent-for-numpy-allclose
rtol=1e-05;
atol=1e-08;
flag = all( abs(a(:)-b(:)) <= atol+rtol*abs(b(:)) );
end
