# functools-ex
## pack
git tag v0.0.1
rm dist/functoolsex-*
python setup.py sdist bdist_wheel
twine upload dist/*
