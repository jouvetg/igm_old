

### <h1 align="center" id="title"> Documentation of print_info </h1>



This serves to print key info on the fly during computation



### <h1 align="center" id="title"> Code of print_info </h1>


```python 

    def print_info(self):
        """
        This serves to print key info on the fly during computation
        """
        if self.saveresult:
            print(
                "IGM %s : Iterations = %6.0f  |  Time = %8.0f  |  DT = %7.2f  |  Ice Volume (km^3) = %10.2f "
                % (
                    datetime.datetime.now().strftime("%H:%M:%S"),
                    self.it,
                    self.t,
                    self.dt_target,
                    np.sum(self.thk) * (self.dx ** 2) / 10 ** 9,
                )
            )

``` 

