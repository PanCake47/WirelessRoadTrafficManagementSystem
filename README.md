# WirelessRoadTrafficManagementSystem
Project for collage assigment.

On start of the program, GTFS files are initialized and a crew index for the current day's trips is built. Then, a loop is executed:

- The API returns the number of vehicles.

- The area filter limits the range to the center.

- For each vehicle, the system searches the index to determine which trips the crew on a given route is running today and what time each departs.

- Calculates the ETA (Estimated Time of Arrival) and delay.

- If the delay and ETA exceed the specified thresholds, the system assigns HIGH or MEDIUM priority, depending on the threshold exceeded.
