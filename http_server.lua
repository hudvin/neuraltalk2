local turbo = require("turbo")

local md5 = require "md5"


require "eval2"
ntp = NTP()

local HelloWorldHandler = class("HelloWorldHandler", turbo.web.RequestHandler)
local ImageHandler = class("ImageHandler", turbo.web.RequestHandler)

function HelloWorldHandler:get()
  self:write("Hello World!")
end

function ImageHandler:post()
  local file_content = self:get_argument("file")

  hash = md5.sumhexa(file_content)
  file_name = "tmp/".. hash
  

  print(file_name)
  output_file = io.open(file_name,"a")
  output_file:write(file_content)
  output_file:close()
  
  filetype_str = io.popen("file "..file_name):read("*a")
  print(filetype_str)
  if string.find(filetype_str, "JPEG") then
     ext = "jpg"
  end
  if string.find(filetype_str, "PNG") then
    ext = "png"
  end 

  new_file_name = 'tmp/'..hash ..'.' .. ext
  print(new_file_name)
  os.execute("mv ".. file_name .. " " .. new_file_name)  

  label = ntp:get_label(new_file_name)
  print(label)
end

turbo.web.Application({{"/hello", HelloWorldHandler},  {"/analyze", ImageHandler} }):listen(8888)
turbo.ioloop.instance():start()
