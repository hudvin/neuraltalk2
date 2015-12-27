require 'torch'


local X = torch.class('A')

 function X:__init(stuff)
   print(stuff)
   self.stuff = stuff
 end


function X:t()
 print(self.stuff)
end


local a = A("xxxx")
a:t()
