clear all
close all
clc

fs = 48000;
maxA_24 = 1-2^(-23);
maxA_16 = 1-2^(-15);

N = 48;
k = [0:10*N-1].';
x = cos(2*pi/N*k);

stem(k, x)
xlabel('k')
ylabel('x[k]')

audiowrite('sine1k_16Bit.wav', x*maxA_16, fs, 'BitsPerSample', 16)  % wav encoding to 16 Bit integer
audiowrite('sine1k_24Bit.wav', x*maxA_24, fs, 'BitsPerSample', 24)  % wav encoding to 24 Bit integer
audiowrite('sine1k_32Bit.wav', x, fs, 'BitsPerSample', 32)  % since x is double, wav encoding is float32 
audiowrite('sine1k_64Bit.wav', x, fs, 'BitsPerSample', 64)  % since x is double, wav encoding is float64
