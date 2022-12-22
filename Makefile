.PHONY: environment.yml data

data: data/adult.csv

data/adult.csv:
	curl -o $@ -L -s https://www.dropbox.com/s/h0nlmmcxe5n1dde/adult.csv?dl=1

environment.yml:
	conda env export | grep -Ev '^(prefix|name): ' > environment.yml
