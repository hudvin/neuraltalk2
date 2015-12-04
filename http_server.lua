local turbo = require("turbo")

local md5 = require "md5"

local HelloWorldHandler = class("HelloWorldHandler", turbo.web.RequestHandler)
local ImageHandler = class("ImageHandler", turbo.web.RequestHandler)

function HelloWorldHandler:get()
  self:write("Hello World!")
end

function ImageHandler:post()
  local file_content = self:get_argument("file")
  hash = md5.sumhexa(file_content)
  output_file = io.open("tmp/" .. hash,"a")
  output_file:write(file_content)
  output_file:close()
end

turbo.web.Application({{"/hello", HelloWorldHandler},  {"/analyze", ImageHandler} }):listen(8888)
turbo.ioloop.instance():start()
