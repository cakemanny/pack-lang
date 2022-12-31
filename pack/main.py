from pack.interp import main

# This is defined in a separate module, because it seems instance checks
# don't necessarily work fully until a module has completed loading.
if __name__ == '__main__':
    main()
