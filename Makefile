.PHONY: test

test:
	@echo "Testing power flow module"
	julia --project=./ test/hs071.jl
