function samples_float32_to_int16( filename )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    fid = fopen(filename, 'r');
    %ss = fread(fid, [2, Inf], 'int16');
    ss = fread(fid, [1, Inf], 'float32');
    fclose(fid);
    maxss = max(ss);
    minss = min(ss);
    aa = zeros(1, length(ss), 'uint16');
    max_uint16 = intmax('uint16');
    for i = 1:length(aa)
        aa(i) = ((ss(i) - minss) / (maxss - minss)) * max_uint16;
    end
    fid = fopen(strcat(filename, '_converted'), 'w');
    fwrite(fid, aa, 'uint16');
    fclose(fid);
end
