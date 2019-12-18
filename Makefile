.PHONY: test

test:
	@echo "Testing alm module"
	julia --project=./ test/hs071.jl
