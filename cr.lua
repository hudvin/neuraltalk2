local md5 = require "md5"
print(md5.sumhexa("xxxxxxxxxxxxxxxxxxxxxxxxx"))


require "eval2"
local ntp = NTP()

local fd, err = io.open("out.jpg", "r")
local file = fd:read("*all")
local sz = file:len()
print(sz)
print(md5.sumhexa(file))


