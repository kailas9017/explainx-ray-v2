-- Turn the SmallCaps spans produced from \TODO / \RES into yellow-highlighted Word runs.
-- Runs after citeproc, so citations inside a marker are already rendered as [n].
local function esc(s)
  return (s:gsub("&", "&amp;"):gsub("<", "&lt;"):gsub(">", "&gt;"))
end

function SmallCaps(el)
  local text = pandoc.utils.stringify(el.content)
  return pandoc.RawInline("openxml",
    '<w:r><w:rPr><w:b/><w:highlight w:val="yellow"/></w:rPr><w:t xml:space="preserve">'
    .. esc(text) .. '</w:t></w:r>')
end
