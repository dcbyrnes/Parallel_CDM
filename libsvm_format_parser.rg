import "regent"

local c = regentlib.c
local std = terralib.includec("stdlib.h")
local cstring = terralib.includec("string.h")

struct data {
    indices     : uint64[256],
    value       : float[256],
    num_entries : uint64,
    label       : float
}

struct DataFile {
    filename        : &int8,
    num_instances   : uint64,
    --instance        : data[256] 
    instance        : &data 
}

terra DataFile:parse()
    c.printf("%s \n", self.filename) 
    var token_label : &int8
    var token_value : &int8
    var token_index : &int8
    var index_delim = ":"
    var entry_delim = " "
    var pos : &int8
    var f_ = c.fopen(self.filename, "r")
    var str : int8[256]
    c.fgets(str, 256, f_)
    c.fgets(str, 256, f_)
    for i = 0, self.num_instances do
        var len : uint64
        var test : int8[256]
        c.fgets(test, 256, f_)
        --c.printf("Input String: %s \n", test)
        token_label = cstring.strtok_r(test, entry_delim, &pos)
        self.instance[i].label = c.atof(token_label)
        token_index = cstring.strtok_r(nil, index_delim, &pos)
        token_value = cstring.strtok_r(nil, entry_delim, &pos)
        var counter = 0
        while (token_value ~= nil) do
            if (token_index ~= nil) then
                -- Convert to 0-indexed format.
                self.instance[i].indices[counter] = (c.atoi(token_index)-1)
            end
            if (token_value ~= nil) then
                self.instance[i].value[counter] = c.atof(token_value)
            end
            counter = counter + 1
            token_index = cstring.strtok_r(nil, index_delim, &pos)
            token_value = cstring.strtok_r(nil, entry_delim, &pos)
        end
        self.instance[i].num_entries = counter
    end
end
return DataFile
